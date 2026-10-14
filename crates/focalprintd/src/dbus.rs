//! The `net.reactivated.Fprint` D-Bus API, as defined by fprintd, so that
//! `pam_fprintd`, `fprintd-*` tools, hyprlock and desktop settings work
//! unchanged.

use crate::{
    sensor::{self, Event, Job, Sensor},
    store::{self, Store},
};
use futures_util::StreamExt as _;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tokio::sync::{Mutex, mpsc};
use zbus::{
    Connection, DBusError, fdo, interface,
    message::Header,
    names::{BusName, OwnedUniqueName, UniqueName},
    object_server::SignalEmitter,
    proxy,
    zvariant::{OwnedObjectPath, Value},
};

pub(crate) const BUS_NAME: &str = "net.reactivated.Fprint";
pub(crate) const MANAGER_PATH: &str = "/net/reactivated/Fprint/Manager";
pub(crate) const DEVICE_PATH: &str = "/net/reactivated/Fprint/Device/0";

/// Touches collected per enrolment.
const ENROLL_STAGES: usize = 20;
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // 20
const ENROLL_STAGES_I32: i32 = ENROLL_STAGES as i32;

#[derive(Debug, DBusError)]
#[zbus(prefix = "net.reactivated.Fprint.Error")]
pub(crate) enum Error {
    #[zbus(error)]
    ZBus(zbus::Error),
    ClaimDevice(String),
    AlreadyInUse(String),
    Internal(String),
    PermissionDenied(String),
    NoEnrolledPrints(String),
    NoActionInProgress(String),
    InvalidFingername(String),
    NoSuchDevice(String),
}

impl From<fdo::Error> for Error {
    fn from(e: fdo::Error) -> Self {
        Self::ZBus(e.into())
    }
}

type Result<T> = std::result::Result<T, Error>;

fn internal(e: impl std::fmt::Display) -> Error {
    tracing::error!(%e, "internal error");
    Error::Internal(e.to_string())
}

#[derive(Debug, Clone, Copy)]
enum Permission {
    Verify,
    Enroll,
    SetUsername,
}

impl Permission {
    fn action(self) -> &'static str {
        match self {
            Self::Verify => "net.reactivated.fprint.device.verify",
            Self::Enroll => "net.reactivated.fprint.device.enroll",
            Self::SetUsername => "net.reactivated.fprint.device.setusername",
        }
    }
}

#[proxy(
    interface = "org.freedesktop.PolicyKit1.Authority",
    default_service = "org.freedesktop.PolicyKit1",
    default_path = "/org/freedesktop/PolicyKit1/Authority"
)]
trait Authority {
    fn check_authorization(
        &self,
        subject: &(&str, HashMap<&str, Value<'_>>),
        action_id: &str,
        details: HashMap<&str, &str>,
        flags: u32,
        cancellation_id: &str,
    ) -> zbus::Result<(bool, bool, HashMap<String, String>)>;
}

/// polkit `CheckAuthorization` flag: let an agent ask for a password.
const ALLOW_USER_INTERACTION: u32 = 1;

#[derive(Debug)]
struct Claim {
    sender: OwnedUniqueName,
    user: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Verify,
    Enroll(&'static str),
}

#[derive(Debug)]
struct Action {
    kind: Kind,
    id: u64,
    cancel: Arc<AtomicBool>,
    /// The final status has been sent; only `*Stop` remains.
    done: bool,
}

#[derive(Debug, Default)]
struct State {
    claim: Option<Claim>,
    action: Option<Action>,
    next_id: u64,
    finger_present: bool,
    finger_needed: bool,
}

/// Shared by the D-Bus objects and the background tasks.
#[derive(Debug)]
pub(crate) struct Daemon {
    store: Store,
    sensor: Sensor,
    state: Mutex<State>,
    last_activity: std::sync::Mutex<Instant>,
}

impl Daemon {
    pub(crate) fn new(store: Store, sensor: Sensor) -> Arc<Self> {
        Arc::new(Self {
            store,
            sensor,
            state: Mutex::default(),
            last_activity: std::sync::Mutex::new(Instant::now()),
        })
    }

    fn touch(&self) {
        if let Ok(mut t) = self.last_activity.lock() {
            *t = Instant::now();
        }
    }

    /// Idle: nobody holds the device and nothing happened for a while.
    pub(crate) async fn idle_for(&self) -> Option<std::time::Duration> {
        if self.state.lock().await.claim.is_some() {
            return None;
        }
        self.last_activity.lock().ok().map(|t| t.elapsed())
    }

    /// Drops the claim of a client that left the bus without releasing.
    pub(crate) async fn client_vanished(&self, name: &UniqueName<'_>) {
        let mut state = self.state.lock().await;
        if state
            .claim
            .as_ref()
            .is_some_and(|c| c.sender.as_str() == name.as_str())
        {
            tracing::info!(client = %name, "claiming client vanished; releasing device");
            if let Some(action) = state.action.take() {
                action.cancel.store(true, Ordering::Relaxed);
            }
            state.claim = None;
        }
    }
}

pub(crate) struct Manager;

// D-Bus methods and properties take `&self` by protocol, used or not.
#[allow(clippy::unused_self)]
#[interface(name = "net.reactivated.Fprint.Manager")]
impl Manager {
    fn get_devices(&self) -> Vec<OwnedObjectPath> {
        vec![device_path()]
    }

    fn get_default_device(&self) -> OwnedObjectPath {
        device_path()
    }
}

fn device_path() -> OwnedObjectPath {
    OwnedObjectPath::try_from(DEVICE_PATH).expect("valid object path")
}

pub(crate) struct Device {
    pub(crate) daemon: Arc<Daemon>,
}

impl Device {
    fn sender(header: &Header<'_>) -> Result<OwnedUniqueName> {
        header
            .sender()
            .map(|s| s.to_owned().into())
            .ok_or_else(|| Error::Internal("message without sender".into()))
    }

    async fn caller_user(conn: &Connection, sender: &UniqueName<'_>) -> Result<String> {
        let uid = fdo::DBusProxy::new(conn)
            .await?
            .get_connection_unix_user(BusName::Unique(sender.clone()))
            .await?;
        let user = nix::unistd::User::from_uid(uid.into()).map_err(internal)?;
        user.map(|u| u.name)
            .ok_or_else(|| Error::Internal(format!("no account for uid {uid}")))
    }

    async fn authorize(
        conn: &Connection,
        sender: &UniqueName<'_>,
        permission: Permission,
    ) -> Result<()> {
        let authority = AuthorityProxy::new(conn).await?;
        let subject = (
            "system-bus-name",
            HashMap::from([("name", Value::from(sender.as_str()))]),
        );
        let (granted, _, _) = authority
            .check_authorization(
                &subject,
                permission.action(),
                HashMap::new(),
                ALLOW_USER_INTERACTION,
                "",
            )
            .await?;
        if granted {
            Ok(())
        } else {
            Err(Error::PermissionDenied(format!(
                "not authorized for {}",
                permission.action()
            )))
        }
    }

    /// The user an operation applies to: the caller, or (with
    /// `setusername`) someone else.
    async fn target_user(
        conn: &Connection,
        sender: &UniqueName<'_>,
        requested: &str,
    ) -> Result<String> {
        let caller = Self::caller_user(conn, sender).await?;
        if requested.is_empty() || requested == caller {
            return Ok(caller);
        }
        Self::authorize(conn, sender, Permission::SetUsername).await?;
        nix::unistd::User::from_name(requested)
            .map_err(internal)?
            .map(|u| u.name)
            .ok_or_else(|| Error::Internal(format!("no such user {requested:?}")))
    }

    /// The user of the caller's claim; fails unless the caller holds it.
    async fn claimed_user(&self, sender: &UniqueName<'_>) -> Result<String> {
        match &self.daemon.state.lock().await.claim {
            Some(c) if c.sender.as_str() == sender.as_str() => Ok(c.user.clone()),
            Some(_) => Err(Error::AlreadyInUse(
                "device is claimed by another client".into(),
            )),
            None => Err(Error::ClaimDevice("device was not claimed".into())),
        }
    }

    /// Starts `job` as the current action and pumps its events to D-Bus.
    async fn start(&self, conn: &Connection, kind: Kind, user: String, job: Job) -> Result<()> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut state = self.daemon.state.lock().await;
        if state.action.is_some() {
            return Err(Error::AlreadyInUse(
                "an action is already in progress".into(),
            ));
        }
        let id = state.next_id;
        state.next_id += 1;
        let cancel = self.daemon.sensor.submit(job, tx);
        state.action = Some(Action {
            kind,
            id,
            cancel,
            done: false,
        });
        drop(state);
        tokio::spawn(pump(conn.clone(), self.daemon.clone(), id, kind, user, rx));
        Ok(())
    }

    async fn stop(&self, sender: &UniqueName<'_>, kind: fn(Kind) -> bool) -> Result<()> {
        self.claimed_user(sender).await?;
        let mut state = self.daemon.state.lock().await;
        match state.action.take_if(|a| kind(a.kind)) {
            Some(action) => {
                action.cancel.store(true, Ordering::Relaxed);
                Ok(())
            }
            None => Err(Error::NoActionInProgress(
                "no such action in progress".into(),
            )),
        }
    }
}

#[allow(clippy::unused_self)] // see `Manager`
#[interface(name = "net.reactivated.Fprint.Device")]
impl Device {
    async fn list_enrolled_fingers(
        &self,
        username: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
    ) -> Result<Vec<String>> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = Self::target_user(conn, &sender, username).await?;
        Self::authorize(conn, &sender, Permission::Verify).await?;
        let fingers = self.daemon.store.list(&user).map_err(internal)?;
        if fingers.is_empty() {
            return Err(Error::NoEnrolledPrints(format!(
                "no prints enrolled for {user}"
            )));
        }
        Ok(fingers.into_iter().map(String::from).collect())
    }

    async fn delete_enrolled_fingers(
        &self,
        username: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
    ) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = Self::target_user(conn, &sender, username).await?;
        Self::authorize(conn, &sender, Permission::Enroll).await?;
        self.daemon.store.delete_all(&user).map_err(internal)
    }

    async fn delete_enrolled_fingers2(
        &self,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
    ) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = self.claimed_user(&sender).await?;
        Self::authorize(conn, &sender, Permission::Enroll).await?;
        self.daemon.store.delete_all(&user).map_err(internal)
    }

    async fn delete_enrolled_finger(
        &self,
        finger_name: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
    ) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = self.claimed_user(&sender).await?;
        if !store::is_finger(finger_name) {
            return Err(Error::InvalidFingername(finger_name.into()));
        }
        Self::authorize(conn, &sender, Permission::Enroll).await?;
        if !self
            .daemon
            .store
            .list(&user)
            .map_err(internal)?
            .contains(&finger_name)
        {
            return Err(Error::NoEnrolledPrints(format!(
                "{finger_name} is not enrolled"
            )));
        }
        self.daemon
            .store
            .delete(&user, finger_name)
            .map_err(internal)
    }

    async fn claim(
        &self,
        username: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
    ) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = Self::target_user(conn, &sender, username).await?;
        if Self::authorize(conn, &sender, Permission::Verify)
            .await
            .is_err()
        {
            Self::authorize(conn, &sender, Permission::Enroll).await?;
        }
        let mut state = self.daemon.state.lock().await;
        if state.claim.is_some() {
            return Err(Error::AlreadyInUse("device already claimed".into()));
        }
        tracing::info!(client = %sender, user, "claimed");
        state.claim = Some(Claim { sender, user });
        Ok(())
    }

    async fn release(&self, #[zbus(header)] header: Header<'_>) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        self.claimed_user(&sender).await?;
        let mut state = self.daemon.state.lock().await;
        if let Some(action) = state.action.take() {
            action.cancel.store(true, Ordering::Relaxed);
        }
        state.claim = None;
        tracing::info!(client = %sender, "released");
        Ok(())
    }

    async fn verify_start(
        &self,
        finger_name: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = self.claimed_user(&sender).await?;
        Self::authorize(conn, &sender, Permission::Verify).await?;
        let mut candidates: Vec<(&'static str, std::sync::Arc<store::Template>)> = self
            .daemon
            .store
            .load_all(&user)
            .map_err(internal)?
            .into_iter()
            .map(|(f, t)| (f, std::sync::Arc::new(t)))
            .collect();
        let selected = if finger_name.is_empty() || finger_name == "any" {
            "any"
        } else if store::is_finger(finger_name) {
            candidates.retain(|(f, _)| *f == finger_name);
            finger_name
        } else {
            return Err(Error::InvalidFingername(finger_name.into()));
        };
        if candidates.is_empty() {
            return Err(Error::NoEnrolledPrints(format!(
                "no prints enrolled for {user}"
            )));
        }
        self.start(
            conn,
            Kind::Verify,
            user.clone(),
            Job::Verify { user, candidates },
        )
        .await?;
        Self::verify_finger_selected(&emitter, selected).await?;
        Ok(())
    }

    async fn verify_stop(&self, #[zbus(header)] header: Header<'_>) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        self.stop(&sender, |k| k == Kind::Verify).await
    }

    async fn enroll_start(
        &self,
        finger_name: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
    ) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        let user = self.claimed_user(&sender).await?;
        let Some(&finger) = store::FINGERS.iter().find(|f| **f == finger_name) else {
            return Err(Error::InvalidFingername(finger_name.into()));
        };
        Self::authorize(conn, &sender, Permission::Enroll).await?;
        let job = Job::Enroll {
            user: user.clone(),
            finger,
            stages: ENROLL_STAGES,
        };
        self.start(conn, Kind::Enroll(finger), user, job).await
    }

    async fn enroll_stop(&self, #[zbus(header)] header: Header<'_>) -> Result<()> {
        self.daemon.touch();
        let sender = Self::sender(&header)?;
        self.stop(&sender, |k| matches!(k, Kind::Enroll(_))).await
    }

    #[zbus(signal)]
    async fn verify_finger_selected(
        emitter: &SignalEmitter<'_>,
        finger_name: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn verify_status(
        emitter: &SignalEmitter<'_>,
        result: &str,
        done: bool,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn enroll_status(
        emitter: &SignalEmitter<'_>,
        result: &str,
        done: bool,
    ) -> zbus::Result<()>;

    #[zbus(property(emits_changed_signal = "const"), name = "name")]
    fn name(&self) -> &'static str {
        "FocalTech FT9368 (FTE4800)"
    }

    #[zbus(property(emits_changed_signal = "const"), name = "num-enroll-stages")]
    fn num_enroll_stages(&self) -> i32 {
        ENROLL_STAGES_I32
    }

    #[zbus(property(emits_changed_signal = "const"), name = "scan-type")]
    fn scan_type(&self) -> &'static str {
        "press"
    }

    #[zbus(property, name = "finger-present")]
    async fn finger_present(&self) -> bool {
        self.daemon.state.lock().await.finger_present
    }

    #[zbus(property, name = "finger-needed")]
    async fn finger_needed(&self) -> bool {
        self.daemon.state.lock().await.finger_needed
    }
}

/// Forwards one job's events: properties always, statuses only while the
/// job is still the current, unfinished action.
async fn pump(
    conn: Connection,
    daemon: Arc<Daemon>,
    id: u64,
    kind: Kind,
    user: String,
    mut events: mpsc::UnboundedReceiver<Event>,
) {
    let iface = match conn
        .object_server()
        .interface::<_, Device>(DEVICE_PATH)
        .await
    {
        Ok(iface) => iface,
        Err(err) => return tracing::error!(%err, "device object missing"),
    };
    let emitter = iface.signal_emitter();
    while let Some(event) = events.recv().await {
        let device = iface.get().await;
        let result = match event {
            Event::FingerPresent(v) => {
                daemon.state.lock().await.finger_present = v;
                device.finger_present_changed(emitter).await
            }
            Event::FingerNeeded(v) => {
                daemon.state.lock().await.finger_needed = v;
                device.finger_needed_changed(emitter).await
            }
            Event::Verify(status, done) if kind == Kind::Verify => {
                if finish(&daemon, id, done).await {
                    Device::verify_status(emitter, status, done).await
                } else {
                    Ok(())
                }
            }
            Event::Enroll(status, done) if matches!(kind, Kind::Enroll(_)) => {
                if finish(&daemon, id, done).await {
                    Device::enroll_status(emitter, status, done).await
                } else {
                    Ok(())
                }
            }
            Event::Enrolled(frames) => {
                let Kind::Enroll(finger) = kind else { continue };
                let template = store::Template::enrolled(frames);
                let status = match daemon.store.save(&user, finger, &template) {
                    Ok(()) => {
                        tracing::info!(user, finger, touches = template.frames.len(), "enrolled");
                        // compute its features now, not at the first unlock
                        let (tx, _rx) = mpsc::unbounded_channel();
                        daemon.sensor.submit(
                            Job::Warm {
                                templates: vec![std::sync::Arc::new(template)],
                            },
                            tx,
                        );
                        "enroll-completed"
                    }
                    Err(err) => {
                        tracing::error!(%err, "cannot store template");
                        "enroll-failed"
                    }
                };
                if finish(&daemon, id, true).await {
                    Device::enroll_status(emitter, status, true).await
                } else {
                    Ok(())
                }
            }
            Event::Adapted {
                finger,
                base,
                template,
            } => {
                // only on top of the template it grew from: a re-enrolment,
                // deletion or other adaptation in between wins
                match daemon.store.load(&user, finger) {
                    Ok(current) if sensor::frames_id(&current.frames) == base => {
                        if let Err(err) = daemon.store.save(&user, finger, &template) {
                            tracing::warn!(%err, user, finger, "cannot store adapted template");
                        }
                    }
                    _ => tracing::info!(user, finger, "template changed meanwhile; not adapting"),
                }
                Ok(())
            }
            Event::Verify(..) | Event::Enroll(..) => Ok(()),
        };
        if let Err(err) = result {
            tracing::warn!(%err, "cannot emit signal");
        }
    }
}

/// Whether action `id` is still current and unfinished; marks it finished
/// when `done`.
async fn finish(daemon: &Daemon, id: u64, done: bool) -> bool {
    let mut state = daemon.state.lock().await;
    match state.action.as_mut() {
        Some(action) if action.id == id && !action.done => {
            action.done = done;
            true
        }
        _ => false,
    }
}

/// Releases claims of clients that disconnect without calling `Release`.
pub(crate) async fn watch_clients(conn: Connection, daemon: Arc<Daemon>) -> zbus::Result<()> {
    let bus = fdo::DBusProxy::new(&conn).await?;
    let mut changes = bus.receive_name_owner_changed().await?;
    while let Some(change) = changes.next().await {
        let Ok(args) = change.args() else { continue };
        if let (BusName::Unique(name), None) = (args.name(), args.new_owner().as_ref()) {
            daemon.client_vanished(name).await;
        }
    }
    Ok(())
}
