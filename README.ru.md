# focalprint

[English](README.md) · **Русский**

Открытый стек для сканеров отпечатков FocalTech в ноутбуках TECNO, на Rust,
без драйверов, библиотек и матчеров производителя
([как это сделано](docs/clean-room.md)).

| Сенсор | Шина | Ноутбук | Статус |
|---|---|---|---|
| FTE4800 / FT9368 | SPI (ACPI) | TECNO MEGABOOK T1 | работает: съёмка, запись, проверка, `sudo` и экран блокировки через API fprintd |
| FT9201 | USB `2808:c652` | TECNO MegaBook K16S | работает: съёмка, запись, проверка, разблокировка; проверено на K16S под Fedora ([инструкция для Fedora](docs/fedora-ft9201.md)) |

Полностью заменяет libfprint + fprintd: `focalprintd` предоставляет D-Bus API
fprintd (`net.reactivated.Fprint`), поэтому `pam_fprintd`, hyprlock, GNOME
и KDE работают без изменений.

> **Бета.** Не отключайте вход по паролю. Качество распознавания ещё
> измеряется (см. [Матчер](#матчер)).

## Структура

| Путь | Что |
|---|---|
| `kernel/` | Модуль ядра на Rust для SPI-сенсора: `/dev/focaltech-fp`, сырые транзакции, сброс, ожидание прерывания ([ABI](docs/protocols.md#ft9368-over-spi-acpi-fte4800)) |
| `crates/focalprint-sensor` | Протоколы сенсоров: FT9368 через модуль, FT9201 через usbfs (`nusb`), общий трейт `Sensor` |
| `crates/focalprintd` | Демон с D-Bus API fprintd, зашифрованное хранилище шаблонов |
| `crates/focalprint-net` | Нейросетевой матчер (FpNet в ONNX через `tract`) и плотное сравнение после совмещения |
| `crates/focalprint-dataset` | Зашифрованный сбор касаний для исследований, только по явному включению |
| `crates/focalprint` | Диагностика: `info`, `capture`, `identify`, `neural`, `dataset` … |
| `crates/focalprint-match` | Ранний классический матчер (ключевые точки + RANSAC), оставлен для сравнения |
| `research/` | Python/PyTorch: обучение FpNet, оценка, синтетические отпечатки, дистилляция, экспорт в ONNX |
| `models/` | Готовая модель FpNet, обученная без чужих (приватных) кадров, скачивается из [релиза](https://github.com/anatoliiii/focalprint/releases/tag/v0.1.0) ([происхождение, источники данных и качество](models/README.md)) |
| `tools/` | `dev-switch` (переключить сенсор на этот стек и обратно), помощник для тулчейна ядра, граббер кадров FT9201 |

## Приватность и безопасность

Кадры отпечатков — биометрия, и обращаются с ними как с секретом:

- кадры живут только в памяти демона, которая заблокирована от свопа
  (`mlockall`), не попадает в core dump и затирается при освобождении
  (`zeroize`), включая буферы USB и SPI;
- кадры и логи не пересекаются: `Debug` у `Image` не печатает пиксели;
- шаблоны шифруются XChaCha20-Poly1305 ключом, запечатанным в TPM через
  `systemd-creds` (`LoadCredentialEncrypted=`), с привязкой к пользователю и
  пальцу (associated data);
- сбор касаний для исследований выключен, пока не включён командой
  `focalprint dataset enable`, и шифруется так же;
- `focalprint dump` пишет только в `/dev/shm` и нужен для разработки.

Модели, обученные на чьих-либо отпечатках, здесь не публикуются.
`fpnet-distill-kd.onnx` не видела ни одного кадра реального человека: она
училась на публичных датасетах и синтетике, а дистилляция от приватно
обученного учителя шла на целях, посчитанных только по публичным картинкам
([models/README.md](models/README.md)).

## Сборка

Пользовательская часть (stable Rust ≥ 1.90):

```sh
cargo build --release            # FT9368 + FT9201
cargo build --release --no-default-features   # без поддержки USB
cargo test --workspace
```

Модуль ядра (нужен только SPI-сенсору). Внешние модули на Rust собираются
только тем же `rustc`, которым собрано работающее ядро; `tools/kernel-toolchain`
скачивает подходящие пакеты Arch Linux `rust` и `bindgen` в
`~/.cache/focalprint/toolchain`:

```sh
make -C kernel
```

USB-сенсору модуль не нужен.

## Запуск (бета)

```sh
sudo tools/dev-switch daemon models/public/fpnet-distill-kd.onnx   # SPI: загрузить модуль, запустить focalprintd
sudo tools/dev-switch c                            # вернуть прежний стек
focalprint --backend usb capture                   # просмотр кадров FT9201 в терминале
```

Демон не запускается без ключа, запечатанного в TPM; systemd передаёт его
через `LoadCredentialEncrypted=`. Создаётся один раз:
`sudo focalprint setup --model models/public/fpnet-distill-kd.onnx`.
`dev-switch` рассчитан на MEGABOOK T1; для FT9201 следуйте
[docs/fedora-ft9201.md](docs/fedora-ft9201.md) (шаги подходят и для других
дистрибутивов). Автозапуск: `tools/focalprint-beta.service.in`.

## Матчер

Окно сенсора около 4,5 × 6 мм, в одно касание попадает лишь несколько
минуций, и классические матчеры не справляются. FpNet — небольшая свёрточная
сеть, которая выдаёт плотные дескрипторы; два касания совмещаются RANSAC по
взаимным совпадениям дескрипторов и оцениваются по согласию дескрипторов на
общей площади кожи.

Демону нужна ONNX-модель; `fpnet-distill-kd.onnx` из
[релиза](https://github.com/anatoliiii/focalprint/releases/tag/v0.1.0) готова
к работе (положите её в `models/public/`; начните с `--threshold 0.41`, см.
[models/README.md](models/README.md)). Свою можно обучить через
`research/train.py` на публичных датасетах; `research/distill.py` переносит
приватно обученную модель в такую, которая не видела приватных кадров, а
`research/export_onnx.py` экспортирует чекпойнт для демона. Протокол оценки и
результаты: `research/eval2.py`.

**Адаптивный шаблон.** Уверенное совпадение (балл ≥ `--adapt-score`, 0.5),
которое не почти-копия уже сохранённого касания, добавляется в шаблон
пальца — до 20 касаний сверх записанных при регистрации; когда место
кончается, оно заменяет самое похожее из добавленных. Касания из регистрации
не заменяются никогда, поэтому случайно принятое чужое касание не может
постепенно захватить шаблон. Так шаблон дорастает до всей подушечки пальца,
и уходит большая часть отказов из-за касания участком кожи, который при
регистрации не попал на сенсор.

## Лицензия

| Часть | Лицензия |
|---|---|
| Пользовательская часть: `crates/`, `research/`, `tools/`, `packaging/` | [LGPL-2.1-or-later](LICENSES/LGPL-2.1-or-later.txt) (`LICENSE`) |
| Модуль ядра: `kernel/` | [GPL-2.0-only](LICENSES/GPL-2.0-only.txt), как само ядро Linux |
| Веса модели (файлы релиза) | [CC-BY-4.0](LICENSES/CC-BY-4.0.txt) |

Знания о протоколе FT9368 основаны на открытом драйвере
[libfprint-fte3600](https://github.com/SamSeven777/libfprint-fte3600)
(LGPL-2.1-or-later); код оттуда не копировался, подробнее в
[docs/clean-room.md](docs/clean-room.md).
Публичные датасеты отпечатков, на которых обучалась модель, здесь не
распространяются и остаются под своими условиями; источники и условия
перечислены в [models/README.md](models/README.md#training-data).
