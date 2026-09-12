import AppKit
import Foundation
import ServiceManagement

if CommandLine.arguments.contains("--unregister-login-item") {
    if #available(macOS 13.0, *) {
        try? SMAppService.mainApp.unregister()
    }
    exit(0)
}

struct Job: Decodable {
    let id: Int
    let original_name: String
    let status: String
    let stage: String
    let progress: Double
    let detail: String?
    let output_dir: String?
}

struct Snapshot: Decodable { let jobs: [Job] }

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let root = Bundle.main.bundleURL.deletingLastPathComponent()
    private var statusItem: NSStatusItem!
    private var watcher: Process?
    private var timer: Timer?
    private let titleItem = NSMenuItem(title: "Ожидание записи", action: nil, keyEquivalent: "")
    private let detailItem = NSMenuItem(title: "Папка inbox пуста", action: nil, keyEquivalent: "")
    private let progress = NSProgressIndicator(frame: NSRect(x: 14, y: 9, width: 218, height: 12))
    private let percent = NSTextField(labelWithString: "0%")

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        if #available(macOS 13.0, *) {
            do {
                if SMAppService.mainApp.status != .enabled {
                    try SMAppService.mainApp.register()
                }
            } catch {
                NSLog("Не удалось зарегистрировать автозапуск: \(error)")
            }
        }
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "🎙 —"
        statusItem.button?.toolTip = "Локальная расшифровка встреч"

        let menu = NSMenu()
        titleItem.isEnabled = false
        detailItem.isEnabled = false
        menu.addItem(titleItem)
        menu.addItem(detailItem)
        let progressItem = NSMenuItem()
        let progressView = NSView(frame: NSRect(x: 0, y: 0, width: 280, height: 31))
        progress.minValue = 0; progress.maxValue = 100; progress.isIndeterminate = false
        percent.frame = NSRect(x: 240, y: 5, width: 38, height: 20)
        percent.alignment = .left
        progressView.addSubview(progress); progressView.addSubview(percent)
        progressItem.view = progressView
        menu.addItem(progressItem)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Показать подробный прогресс…", action: #selector(openDashboard), keyEquivalent: "")
        menu.addItem(withTitle: "Открыть результаты", action: #selector(openOutputs), keyEquivalent: "")
        menu.addItem(withTitle: "Открыть входящие записи", action: #selector(openInbox), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Перезапустить обработчик", action: #selector(restartWatcher), keyEquivalent: "")
        menu.addItem(withTitle: "Завершить", action: #selector(quit), keyEquivalent: "q")
        menu.items.forEach { $0.target = self }
        statusItem.menu = menu

        startWatcher()
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in self?.refresh() }
    }

    private func startWatcher() {
        guard watcher?.isRunning != true else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [root.appendingPathComponent("pipeline.py").path, "watch"]
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        process.environment = environment
        let log = root.appendingPathComponent("state/menu-watcher.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        if let handle = try? FileHandle(forWritingTo: log) {
            handle.seekToEndOfFile(); process.standardOutput = handle; process.standardError = handle
        }
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { self?.startWatcher() }
        }
        do { try process.run(); watcher = process } catch { detailItem.title = "Не удалось запустить обработчик" }
    }

    private func refresh() {
        let url = root.appendingPathComponent("state/progress.json")
        guard let data = try? Data(contentsOf: url), let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data) else {
            statusItem.button?.title = "🎙 —"; return
        }
        let job = snapshot.jobs.first(where: { $0.status == "running" || $0.status == "queued" }) ?? snapshot.jobs.first
        guard let job else { titleItem.title = "Ожидание записи"; progress.doubleValue = 0; percent.stringValue = "0%"; return }
        let value = min(100, max(0, job.progress))
        let rounded = Int(value.rounded())
        progress.doubleValue = value
        percent.stringValue = "\(rounded)%"
        titleItem.title = job.original_name
        detailItem.title = job.detail ?? stageName(job.stage)
        statusItem.button?.title = job.status == "done" ? "🎙 ✓" : "🎙 \(rounded)%"
        statusItem.button?.toolTip = "\(job.original_name) — \(detailItem.title)"
    }

    private func stageName(_ stage: String) -> String {
        ["queued":"В очереди", "validate":"Проверка записи", "extract_audio":"Подготовка аудио", "diarization":"Определение участников", "transcription":"Распознавание речи", "export":"Создание файлов", "done":"Готово"][stage] ?? stage
    }

    @objc private func openDashboard() { NSWorkspace.shared.open(URL(string: "http://127.0.0.1:8765")!) }
    @objc private func openOutputs() { NSWorkspace.shared.open(root.appendingPathComponent("outputs")) }
    @objc private func openInbox() { NSWorkspace.shared.open(root.appendingPathComponent("inbox")) }
    @objc private func restartWatcher() { watcher?.terminate(); watcher = nil; DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.startWatcher() } }
    @objc private func quit() { watcher?.terminate(); NSApp.terminate(nil) }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
