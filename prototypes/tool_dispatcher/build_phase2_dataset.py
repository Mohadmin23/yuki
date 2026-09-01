"""Build and freeze the hand-curated Phase 2 reliability dataset."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .registry import ToolRegistry

OUTPUT_PATH = Path(__file__).with_name("phase2-cases.json")
CHECKSUM_PATH = Path(__file__).with_name("phase2-cases.sha256")

type Seed = tuple[str, str | None]

STANDARD_CATEGORIES = (
    ["normal"] * 11
    + ["paraphrase"] * 6
    + ["confusion"] * 4
    + ["preservation"] * 3
    + ["adversarial"]
)
NO_ARGUMENT_CATEGORIES = (
    ["normal"] * 13
    + ["paraphrase"] * 7
    + ["confusion"] * 4
    + ["adversarial"]
)


CASE_BANKS: dict[str, list[Seed]] = {
    "time": [
        ("What time is it right now?", None),
        ("Tell me today's date and local time.", None),
        ("Check the clock for me.", None),
        ("I need the current day, date, and time.", None),
        ("Give me the local time on this Mac.", None),
        ("What's today's date?", None),
        ("Current timestamp, please.", None),
        ("Can you check the time?", None),
        ("Which day of the week is it?", None),
        ("Read the current clock.", None),
        ("Date and time now.", None),
        ("Before we continue, tell me the current local time.", None),
        ("Quick clock check.", None),
        ("hey, what hour are we at?", None),
        ("CURRENT TIME?", None),
        ("whats the date today", None),
        ("Could I trouble you for whatever the clock says at the moment?", None),
        ("I lost track—day and time?", None),
        ("Need a now-check: calendar date plus clock.", None),
        ("umm... time rn?", None),
        ("Give current time; do not search the web.", None),
        ("Use the clock, not a shell date command.", None),
        ("What's the time here—not Tokyo weather.", None),
        ("Tell me today's date directly; don't ask Claude.", None),
        ("Delegated tool request, very terse: now?", None),
    ],
    "weather": [
        ("What's the weather in Tokyo right now?", "Tokyo"),
        ("Give me current conditions for New York.", "New York"),
        ("Check today's weather in Algiers.", "Algiers"),
        ("How is the weather in St. John's?", "St. John's"),
        ("Current weather for Mexico City, please.", "Mexico City"),
        ("Tell me the conditions in Seoul.", "Seoul"),
        ("Weather check for Cape Town.", "Cape Town"),
        ("What is it like outside in São Paulo?", "São Paulo"),
        ("Get the current weather in Reykjavik.", "Reykjavik"),
        ("How's Auckland's weather today?", "Auckland"),
        ("Please check conditions for Montréal.", "Montréal"),
        ("Is it jacket weather in Glasgow right now?", "Glasgow"),
        ("How's the sky treating folks in Nairobi today?", "Nairobi"),
        ("temperature check—Osaka", "Osaka"),
        ("Rain or shine around Buenos Aires?", "Buenos Aires"),
        ("Could you pull current weather for Ho Chi Minh City?", "Ho Chi Minh City"),
        ("weather pls: Québec City", "Québec City"),
        ("Tell me current weather in Tokyo; don't search for weather APIs.", "Tokyo"),
        ("Use weather for Paris, not fetch on a weather URL.", "Paris"),
        ("What's the weather in Lima—not web articles about its climate?", "Lima"),
        ("Current conditions in Rome; no need to ask Claude.", "Rome"),
        ("Weather for Washington, D.C. exactly as written.", "Washington, D.C."),
        ("Check weather in Xi'an; preserve the apostrophe.", "Xi'an"),
        ("Current weather: São Luís", "São Luís"),
        ("weather... city is 'St. Louis' (keep the period)", "St. Louis"),
    ],
    "fetch": [
        ("Fetch this page: https://example.com", "https://example.com"),
        (
            "Retrieve https://docs.python.org/3/library/pathlib.html",
            "https://docs.python.org/3/library/pathlib.html",
        ),
        ("Fetch http://localhost:8080/status", "http://localhost:8080/status"),
        (
            "Get the content at https://api.example.org/v1/items?id=42",
            "https://api.example.org/v1/items?id=42",
        ),
        (
            "Download the page https://en.wikipedia.org/wiki/Apple_M1",
            "https://en.wikipedia.org/wiki/Apple_M1",
        ),
        ("Fetch https://example.net/a%20b", "https://example.net/a%20b"),
        (
            "Read the web page at https://developer.apple.com/metal/",
            "https://developer.apple.com/metal/",
        ),
        (
            "Retrieve https://huggingface.co/MadeAgents/Hammer2.1-3b",
            "https://huggingface.co/MadeAgents/Hammer2.1-3b",
        ),
        (
            "Fetch https://example.com/report.pdf#page=4",
            "https://example.com/report.pdf#page=4",
        ),
        (
            "Get text from https://sub.example.co.uk/path/to/page",
            "https://sub.example.co.uk/path/to/page",
        ),
        (
            "Fetch https://example.org/?q=yuki&lang=en",
            "https://example.org/?q=yuki&lang=en",
        ),
        (
            "Open this address and bring back its content: https://example.com/about",
            "https://example.com/about",
        ),
        (
            "I already have the link—download its text: https://example.net/news/latest",
            "https://example.net/news/latest",
        ),
        (
            "Use the exact endpoint https://api.example.com/v2/health",
            "https://api.example.com/v2/health",
        ),
        (
            "Please retrieve what's at http://example.org/plain.txt",
            "http://example.org/plain.txt",
        ),
        (
            "Page content needed from https://example.edu/research/index.html",
            "https://example.edu/research/index.html",
        ),
        (
            "Go straight to https://example.com/docs/start—no discovery needed.",
            "https://example.com/docs/start",
        ),
        (
            "Fetch this exact URL; do not search for it: https://example.com/guide",
            "https://example.com/guide",
        ),
        (
            "Use fetch, not weather, on https://weather.example/api?city=Tokyo",
            "https://weather.example/api?city=Tokyo",
        ),
        (
            "This URL discusses cameras; fetch it rather than using vision: https://example.org/camera",
            "https://example.org/camera",
        ),
        (
            "Retrieve the Claude documentation URL, don't ask Claude: https://docs.example.com/claude",
            "https://docs.example.com/claude",
        ),
        (
            "Preserve every character in https://EXAMPLE.com/MixedCase?q=A_B-7#Top",
            "https://EXAMPLE.com/MixedCase?q=A_B-7#Top",
        ),
        (
            "Fetch exactly https://example.com/a%2Fb?next=%2Fhome%3Fx%3D1",
            "https://example.com/a%2Fb?next=%2Fhome%3Fx%3D1",
        ),
        (
            "Use this signed-looking URL unchanged: https://cdn.example.net/file?X-Amz-Date=20260819T120000Z&X-Amz-Signature=aB_9-xy",
            "https://cdn.example.net/file?X-Amz-Date=20260819T120000Z&X-Amz-Signature=aB_9-xy",
        ),
        (
            "Fetch it—even with odd punctuation: https://example.com/a(b)?x=1,2",
            "https://example.com/a(b)?x=1,2",
        ),
    ],
    "search": [
        ("Search the web for the latest MLX documentation.", "latest MLX documentation"),
        ("Find information about Apple Silicon unified memory.", "Apple Silicon unified memory"),
        ("Search for current Python 3.14 release notes.", "current Python 3.14 release notes"),
        ("Look up Hammer 2.1 function calling results.", "Hammer 2.1 function calling results"),
        ("Web search: best local speech synthesis models", "best local speech synthesis models"),
        ("Find recent articles about JSON schema decoding.", "recent articles about JSON schema decoding"),
        ("Search online for M1 memory pressure benchmarks.", "M1 memory pressure benchmarks"),
        ("Look up how APFS handles case-sensitive paths.", "how APFS handles case-sensitive paths"),
        ("Search for open-source tool-calling datasets.", "open-source tool-calling datasets"),
        ("Find the official Rust async book.", "official Rust async book"),
        ("Search the internet for local LLM routing research.", "local LLM routing research"),
        ("Can you dig up recent notes on MLX-LM streaming?", "recent notes on MLX-LM streaming"),
        ("I need web results for compact function-calling models.", "compact function-calling models"),
        ("See what the internet says about Metal GPU utilization.", "Metal GPU utilization"),
        ("Track down documentation for Python pathlib expanduser.", "Python pathlib expanduser"),
        ("Could you research deterministic JSON generation locally?", "deterministic JSON generation locally"),
        ("google-ish lookup: macOS unified memory swap behavior", "macOS unified memory swap behavior"),
        ("Search for Tokyo weather APIs, not Tokyo's current weather.", "Tokyo weather APIs"),
        ("Find a website explaining URL fragments; there is no exact URL to fetch.", "website explaining URL fragments"),
        ("Search the web for camera calibration guides; don't open the camera.", "camera calibration guides"),
        ("Look up Claude tool-use documentation; do not delegate to Claude.", "Claude tool-use documentation"),
        ("Search exactly for: C++ std::filesystem path case-sensitivity", "C++ std::filesystem path case-sensitivity"),
        (
            "Keep this query intact: site:huggingface.co \"Hammer2.1-3b\" function calling",
            "site:huggingface.co \"Hammer2.1-3b\" function calling",
        ),
        (
            "Use the exact query foo_bar() \"KeyError: x-y\"",
            "foo_bar() \"KeyError: x-y\"",
        ),
        ("Search, punctuation and all: ??? macOS swap usage M1 -- 2026", "??? macOS swap usage M1 -- 2026"),
    ],
    "calc": [
        ("Calculate 27 + 58.", "27 + 58"),
        ("What is 144 / 12?", "144 / 12"),
        ("Compute (19 * 7) - 8.", "(19 * 7) - 8"),
        ("Evaluate 81 % 7.", "81 % 7"),
        ("Calculate 3.5 * 2.4.", "3.5 * 2.4"),
        ("Work out (200 - 17) / 3.", "(200 - 17) / 3"),
        ("Compute 9 * (6 + 4).", "9 * (6 + 4)"),
        ("What's 1000 / 16?", "1000 / 16"),
        ("Evaluate (42 + 18) % 11.", "(42 + 18) % 11"),
        ("Calculate 0.75 + 1.125.", "0.75 + 1.125"),
        ("Find the value of ((8 * 8) - 4) / 5.", "((8 * 8) - 4) / 5"),
        ("number crunch this for me: 17*23", "17*23"),
        ("Quick maths—(96/8)+13", "(96/8)+13"),
        ("How much do you get if 5.2 is multiplied by 9?", "5.2 * 9"),
        ("Resolve this arithmetic: 400 - (17 * 6)", "400 - (17 * 6)"),
        ("gimme the result of 11 % 4", "11 % 4"),
        ("Please do the numbers: (2 + 3) * (7 - 1).", "(2 + 3) * (7 - 1)"),
        ("Calculate 144 / 12; don't search for the answer.", "144 / 12"),
        ("Use the calculator, not a shell expression: 7 * 13.", "7 * 13"),
        ("This mentions RAM, but just compute 16 * 1024.", "16 * 1024"),
        ("Don't ask Claude—evaluate (55 - 9) / 2.", "(55 - 9) / 2"),
        ("Preserve the expression exactly: (144 / 12) + 7", "(144 / 12) + 7"),
        ("Calculate this spacing-sensitive input:  8 * (3 + 2) ", "8 * (3 + 2)"),
        ("Evaluate exactly 10.00/4.0", "10.00/4.0"),
        ("tool delegation is messy, but the arithmetic is ((1+2)*(3+4))%5", "((1+2)*(3+4))%5"),
    ],
    "hardware": [
        ("How much RAM is this Mac using?", "ram"),
        ("Show me current CPU usage.", "cpu"),
        ("Check disk utilization on this computer.", "disk"),
        ("Give me all available hardware statistics.", "all"),
        ("Which processes are consuming resources?", "process"),
        ("Show the top resource-using processes.", "top"),
        ("Check the local GPU status.", "gpu"),
        ("Read the machine temperature.", "temp"),
        ("Report RAM pressure.", "ram"),
        ("I need a CPU status snapshot.", "cpu"),
        ("How full are the local disks?", "disk"),
        ("what's the machine doing memory-wise?", "ram"),
        ("Give me a quick processor-health glance.", "cpu"),
        ("Is storage getting cramped?", "disk"),
        ("Show everything about the Mac's current resource state.", "all"),
        ("Which apps are hogging this machine?", "top"),
        ("how hot is the laptop running rn", "temp"),
        ("Use hardware RAM stats, not recalled conversations about memory.", "ram"),
        ("Check CPU through the hardware tool; don't run top in shell.", "cpu"),
        ("Report disk usage instead of reading a file from disk.", "disk"),
        ("I want GPU status, not an image generation.", "gpu"),
        ("Pass the hardware metric exactly as `process`.", "process"),
        ("Use metric `top` exactly, without turning it into a shell command.", "top"),
        ("Hardware metric must remain `temp`.", "temp"),
        ("Resource check—uh, the one for all stats, not just one.", "all"),
    ],
    "see": [
        ("Use the camera to find my keys.", "keys"),
        ("Describe what the camera can see right now.", None),
        ("Look through Yuki's eye for my coffee mug.", "coffee mug"),
        ("Check the door with the camera.", "door"),
        ("Find my red backpack in the room.", "red backpack"),
        ("Inspect the whiteboard using vision.", "whiteboard"),
        ("Can the webcam spot my phone?", "phone"),
        ("Look around the room and describe the scene.", None),
        ("Locate my charging cable with the camera.", "charging cable"),
        ("Point the camera toward the window and find the window.", "window"),
        ("Use vision to locate my wallet.", "wallet"),
        ("Use your eyes and track down my earbuds.", "earbuds"),
        ("Take a look around—what's visible?", None),
        ("Could vision hunt for the blue notebook?", "blue notebook"),
        ("Peek through the webcam for my glasses.", "glasses"),
        ("Scan the room until you find the desk lamp.", "desk lamp"),
        ("what does yuki's eye see atm", None),
        ("Find the printed image with the camera; do not generate a new image.", "printed image"),
        ("Look at the thermostat display, not an online weather service.", "thermostat display"),
        ("Visually inspect the terminal window; don't execute its command.", "terminal window"),
        ("Find the notes file lying on my desk; do not read a filesystem file.", "notes file"),
        ("Camera target exactly: USB-C adapter", "USB-C adapter"),
        ("Locate `AirPods Pro` and preserve that capitalization.", "AirPods Pro"),
        ("Use vision for the object named M.2 enclosure.", "M.2 enclosure"),
        ("I can't describe it well—camera, find the 'red-ish cable'?", "red-ish cable"),
    ],
    "image": [
        ("Generate an image with this prompt: neon fox beneath a rainy sign", "neon fox beneath a rainy sign"),
        ("Create a picture of a watercolor lighthouse at dawn.", "watercolor lighthouse at dawn"),
        ("Make artwork showing a tiny robot watering sunflowers.", "tiny robot watering sunflowers"),
        ("Generate this scene: moonlit anime library", "moonlit anime library"),
        ("Create an image of a glass city floating above clouds.", "glass city floating above clouds"),
        ("Render a cozy pixel-art kitchen in winter.", "cozy pixel-art kitchen in winter"),
        ("Make a poster of an astronaut playing cello on Mars.", "astronaut playing cello on Mars"),
        ("Generate a charcoal sketch of an old cedar tree.", "charcoal sketch of an old cedar tree"),
        ("Create a cinematic desert train at sunset.", "cinematic desert train at sunset"),
        ("Draw a blue dragon curled around a lighthouse.", "blue dragon curled around a lighthouse"),
        ("Make an image of bioluminescent jellyfish over a dark reef.", "bioluminescent jellyfish over a dark reef"),
        ("Could you illustrate a sleepy cat running a ramen shop?", "sleepy cat running a ramen shop"),
        ("I want a visual: paper-cut mountains under a gold moon.", "paper-cut mountains under a gold moon"),
        ("Turn this idea into artwork—retro-futurist bicycle workshop.", "retro-futurist bicycle workshop"),
        ("Paint, digitally, a storm moving over lavender fields.", "storm moving over lavender fields"),
        ("picture pls: friendly moss-covered forest giant", "friendly moss-covered forest giant"),
        ("Visualize an isometric library inside a teacup.", "isometric library inside a teacup"),
        ("Generate a camera-shaped sculpture; don't use the real camera.", "camera-shaped sculpture"),
        ("Create an image of Tokyo weather icons, not a weather lookup.", "Tokyo weather icons"),
        ("Make poster artwork containing the words notes.md; do not write a file.", "poster artwork containing the words notes.md"),
        ("Illustrate Claude as a brass automaton; do not ask Claude anything.", "Claude as a brass automaton"),
        ("Use this exact image prompt: RED fox, blue_sky, 35mm—high contrast!", "RED fox, blue_sky, 35mm—high contrast!"),
        ("Preserve punctuation in the prompt: 'Home, 2:17 a.m.' in rainy cyberpunk style", "'Home, 2:17 a.m.' in rainy cyberpunk style"),
        ("Generate exactly: café façade; crème-and-teal palette; 4:3 composition", "café façade; crème-and-teal palette; 4:3 composition"),
        ("uhh make... an image? prompt is: almost-empty white room + one RED chair", "almost-empty white room + one RED chair"),
    ],
    "read": [
        ("Read /tmp/report.txt.", "/tmp/report.txt"),
        ("Open ~/Documents/notes.md.", "~/Documents/notes.md"),
        ("Show me the contents of ./config.toml.", "./config.toml"),
        ("Read ../logs/app.log from the filesystem.", "../logs/app.log"),
        ("Open /Users/mohamedlaminemennane/Desktop/Todo.txt.", "/Users/mohamedlaminemennane/Desktop/Todo.txt"),
        ("Read /Volumes/madisk/data/sample.json.", "/Volumes/madisk/data/sample.json"),
        ("Load the local file /private/tmp/yuki test.txt.", "/private/tmp/yuki test.txt"),
        ("Read ./src/main.py.", "./src/main.py"),
        ("Open ~/Library/Application Support/demo/settings.json.", "~/Library/Application Support/demo/settings.json"),
        ("Show the local file /etc/hosts.", "/etc/hosts"),
        ("Read /tmp/.hidden from disk.", "/tmp/.hidden"),
        ("Peek inside the file sitting at ~/Downloads/result.csv.", "~/Downloads/result.csv"),
        ("Could you pull up what's in ./README.md?", "./README.md"),
        ("I need the text stored under /opt/demo/config.yaml.", "/opt/demo/config.yaml"),
        ("Have a look at ../data/events.jsonl.", "../data/events.jsonl"),
        ("local-file check: /var/tmp/service.log", "/var/tmp/service.log"),
        ("Can you inspect ~/Desktop/draft.txt for me?", "~/Desktop/draft.txt"),
        ("Read /tmp/yuki_notes.txt from the normal filesystem, not Yuki's folder.", "/tmp/yuki_notes.txt"),
        ("Open ./notes.md directly; don't run `cat` in a shell.", "./notes.md"),
        ("Read /tmp/ram-report.txt; this is a file, not live hardware RAM.", "/tmp/ram-report.txt"),
        ("Use normal file read for /tmp/claude.txt, not ask_claude.", "/tmp/claude.txt"),
        ("Preserve case exactly when reading /tmp/FooBar.JSON.", "/tmp/FooBar.JSON"),
        ("Read the path with spaces: ~/My Notes/Plan v2.md", "~/My Notes/Plan v2.md"),
        ("Open exactly /tmp/café/naïve-file.txt", "/tmp/café/naïve-file.txt"),
        ("Normal filesystem request despite the name: /Users/me/Projects/Yuki/data.JSON", "/Users/me/Projects/Yuki/data.JSON"),
    ],
    "shell": [
        ("Run the shell command pwd.", "pwd"),
        ("Execute ls -la in the terminal.", "ls -la"),
        ("Run whoami.", "whoami"),
        ("Execute the date command.", "date"),
        ("Run uptime in the shell.", "uptime"),
        ("Execute df -h.", "df -h"),
        ("Run uname -a.", "uname -a"),
        ("Execute hostname.", "hostname"),
        ("Run which python.", "which python"),
        ("Use the terminal for wc -l /tmp/demo.txt.", "wc -l /tmp/demo.txt"),
        ("Run head -n 5 /tmp/log.txt.", "head -n 5 /tmp/log.txt"),
        ("terminal, please: tail -n 10 /tmp/app.log", "tail -n 10 /tmp/app.log"),
        ("Ask the command line where I am with `pwd`.", "pwd"),
        ("Have the shell print the machine name using hostname.", "hostname"),
        ("I need a directory listing: ls -lah /tmp", "ls -lah /tmp"),
        ("Command-line check for python3's location: which python3", "which python3"),
        ("please terminal this exact command: wc -c ./README.md", "wc -c ./README.md"),
        ("Use shell `cat /tmp/report.txt`, not the read tool.", "cat /tmp/report.txt"),
        ("Run `date` as a command rather than using Yuki's time tool.", "date"),
        ("Use `df -h` in shell, not the hardware summary.", "df -h"),
        ("List Yuki-named paths using `ls -la yuki`, not yuki_list.", "ls -la yuki"),
        ("Preserve this command exactly: echo \"Yuki_2.0\"", "echo \"Yuki_2.0\""),
        ("Run exactly cat \"/tmp/My Notes.txt\"", "cat \"/tmp/My Notes.txt\""),
        ("Keep command case and path: tail -n 20 ./logs/App.LOG", "tail -n 20 ./logs/App.LOG"),
        ("The delegation is noisy [shell??], but run: ls -la /tmp/does-not-exist", "ls -la /tmp/does-not-exist"),
    ],
    "yuki_write": [
        ("Write this into Yuki's file: todo.txt|buy oat milk", "todo.txt|buy oat milk"),
        ("Create Yuki note ideas.md|build a tiny weather station", "ideas.md|build a tiny weather station"),
        ("Save in Yuki's folder journal.txt|Walked by the river today.", "journal.txt|Walked by the river today."),
        ("Make a Yuki file colors.txt|teal, amber, violet", "colors.txt|teal, amber, violet"),
        ("Write recipe.md|Add cardamom after the water boils to Yuki files.", "recipe.md|Add cardamom after the water boils"),
        ("Create Yuki file project.log|prototype started at 09:30", "project.log|prototype started at 09:30"),
        ("Store draft.txt|The moon was low over the harbor. in Yuki's folder.", "draft.txt|The moon was low over the harbor."),
        ("Write Yuki-local file links.txt|https://example.com/start", "links.txt|https://example.com/start"),
        ("Create goals.md|Finish the dispatcher benchmark by Friday in Yuki files.", "goals.md|Finish the dispatcher benchmark by Friday"),
        ("Save quote.txt|Small steps still cross mountains. as a Yuki file.", "quote.txt|Small steps still cross mountains."),
        ("Write status.json|{\"ready\": true} in Yuki's personal folder.", "status.json|{\"ready\": true}"),
        ("Jot this down in Yuki space—snack.txt|almonds and tea", "snack.txt|almonds and tea"),
        ("Start a fresh Yuki note named plan.md containing Call Sam at noon.", "plan.md|Call Sam at noon"),
        ("Put `hello from M1` into machine.txt in Yuki's own files.", "machine.txt|hello from M1"),
        ("Yuki-file creation request: mood.txt|calm but curious", "mood.txt|calm but curious"),
        ("Could you make books.csv in your folder with Dune,Neuromancer?", "books.csv|Dune,Neuromancer"),
        ("new personal file — name: seed.txt; contents: basil on the windowsill", "seed.txt|basil on the windowsill"),
        ("Create fresh Yuki file poem.md|first line; do not append to an old file.", "poem.md|first line"),
        ("Write camera-notes.txt|lens cap missing; don't use vision.", "camera-notes.txt|lens cap missing"),
        ("Save image_prompt.txt|red fox in snow as text, not generated art.", "image_prompt.txt|red fox in snow"),
        ("Write memory.txt|user likes mint as a file, not persistent memory.", "memory.txt|user likes mint"),
        ("Preserve exactly: CaseFile.TXT|Line ONE, line Two!", "CaseFile.TXT|Line ONE, line Two!"),
        ("Yuki write payload exactly code_note.md|foo_bar() -> KeyError: 'x-y'", "code_note.md|foo_bar() -> KeyError: 'x-y'"),
        ("Create `My Plan 2.md` with exact content `Meet @ 7:05 p.m. — café`.", "My Plan 2.md|Meet @ 7:05 p.m. — café"),
        ("write... in Yuki, odd name.txt with value = a_b-c.d?", "odd name.txt|value = a_b-c.d?"),
    ],
    "yuki_read": [
        ("Read Yuki's todo.txt file.", "todo.txt"),
        ("Open notes.md from Yuki's folder.", "notes.md"),
        ("Show me Yuki file journal.txt.", "journal.txt"),
        ("Read recipe.md in Yuki's personal files.", "recipe.md"),
        ("Open the Yuki-local file ideas.txt.", "ideas.txt"),
        ("Read status.json from Yuki's scratch space.", "status.json"),
        ("Show Yuki's poem.md file.", "poem.md"),
        ("Open links.txt in Yuki files.", "links.txt"),
        ("Read the personal Yuki note goals.md.", "goals.md"),
        ("Get the contents of Yuki's quote.txt.", "quote.txt"),
        ("Read Yuki-local config.toml.", "config.toml"),
        ("Pull up your own little file called snack.txt.", "snack.txt"),
        ("What did you put in books.csv in your folder?", "books.csv"),
        ("lemme see the Yuki note mood.txt", "mood.txt"),
        ("Could you open your personal draft.txt?", "draft.txt"),
        ("From your scratch space, show machine.txt.", "machine.txt"),
        ("Yuki-read request, filename is seed.txt.", "seed.txt"),
        ("Read Yuki's notes.md, not ~/Documents/notes.md.", "notes.md"),
        ("Open one Yuki file named index.txt; do not list the folder.", "index.txt"),
        ("Use yuki_read for shell.txt; don't execute its contents.", "shell.txt"),
        ("Read memory.txt as a Yuki file, not recalled memory.", "memory.txt"),
        ("Preserve filename capitalization: CaseFile.TXT", "CaseFile.TXT"),
        ("Read Yuki filename exactly `My Plan 2.md`.", "My Plan 2.md"),
        ("Open the Yuki file café-notes.md with the accent intact.", "café-notes.md"),
        ("personal-file delegation, maybe hidden: read .env.example", ".env.example"),
    ],
    "yuki_list": [
        ("List Yuki's files.", None),
        ("Show every file in Yuki's personal folder.", None),
        ("What files are in Yuki's scratch space?", None),
        ("Give me the Yuki file list.", None),
        ("Display the contents of Yuki's file directory.", None),
        ("List the notes Yuki has saved locally.", None),
        ("Show names of files in the yuki folder.", None),
        ("Check which Yuki-local files exist.", None),
        ("Enumerate Yuki's personal files.", None),
        ("I want a listing of the Yuki workspace files.", None),
        ("Show the current Yuki scratch-file inventory.", None),
        ("List all files saved by Yuki.", None),
        ("Open the Yuki file index—not one specific file.", None),
        ("what little notes do you have in your own folder?", None),
        ("lemme see the filenames in yuki space", None),
        ("Could you inventory your personal scratch directory?", None),
        ("Which files have you got tucked away locally?", None),
        ("show me your file shelf", None),
        ("Yuki folder contents, names only please.", None),
        ("Do a quick personal-files roll call.", None),
        ("Use Yuki file listing, not shell `ls`.", None),
        ("List the Yuki folder; do not read notes.md yet.", None),
        ("Show Yuki files, not the normal filesystem directory.", None),
        ("Inventory saved notes; don't recall conversation memory.", None),
        ("delegated request is vague: your files... what exists?", None),
    ],
    "yuki_delete": [
        ("Delete old.txt from Yuki's files.", "old.txt"),
        ("Remove draft.md from Yuki's folder.", "draft.md"),
        ("Erase Yuki-local temp.json.", "temp.json"),
        ("Delete the personal Yuki file shopping.txt.", "shopping.txt"),
        ("Remove archive.log from Yuki's scratch space.", "archive.log"),
        ("Delete unused.csv in Yuki files.", "unused.csv"),
        ("Erase Yuki's test-note.md.", "test-note.md"),
        ("Remove duplicate.txt from the personal folder.", "duplicate.txt"),
        ("Delete Yuki file obsolete.toml.", "obsolete.toml"),
        ("Get rid of Yuki-local scratch.py.", "scratch.py"),
        ("Delete yesterday.txt from Yuki's saved files.", "yesterday.txt"),
        ("Please toss the file named old-plan.md from your own folder.", "old-plan.md"),
        ("Clear out personal note stale.txt.", "stale.txt"),
        ("I no longer need tmp.csv in Yuki space—remove it.", "tmp.csv"),
        ("Can you discard your saved file retry.log?", "retry.log"),
        ("Yuki file cleanup: bye.md should go.", "bye.md"),
        ("nuke, from your folder only, demo.json", "demo.json"),
        ("Delete Yuki's report.txt; do not delete /tmp/report.txt.", "report.txt"),
        ("Remove one Yuki file named notes.md, not list all files.", "notes.md"),
        ("Delete image.png as a Yuki file; don't generate an image.", "image.png"),
        ("Forget file fact.txt by deleting it, not by editing memory.", "fact.txt"),
        ("Delete filename exactly `OldNotes.MD` from Yuki.", "OldNotes.MD"),
        ("Remove the Yuki file `My Draft 2.txt` with spaces intact.", "My Draft 2.txt"),
        ("Erase café-list.txt; keep the accented filename.", "café-list.txt"),
        ("Yuki cleanup request, despite uncertainty: delete .old-config", ".old-config"),
    ],
    "yuki_append": [
        ("Append to Yuki file todo.txt|call the dentist", "todo.txt|call the dentist"),
        ("Add to journal.md|Saw two herons by the lake. in Yuki's folder.", "journal.md|Saw two herons by the lake."),
        ("Append Yuki note ideas.txt|try a paper prototype", "ideas.txt|try a paper prototype"),
        ("Extend the personal file log.txt|second benchmark started", "log.txt|second benchmark started"),
        ("Add one line to Yuki's books.txt|The Left Hand of Darkness", "books.txt|The Left Hand of Darkness"),
        ("Append recipe.md|Use less sugar next time in Yuki files.", "recipe.md|Use less sugar next time"),
        ("Extend Yuki-local links.md|https://example.org/new", "links.md|https://example.org/new"),
        ("Append to goals.txt|ship the small prototype", "goals.txt|ship the small prototype"),
        ("Add to Yuki's colors.csv|indigo,#4B0082", "colors.csv|indigo,#4B0082"),
        ("Append status.log|phase two queued to the Yuki file.", "status.log|phase two queued"),
        ("Extend the personal poem.md|and morning found the shore", "poem.md|and morning found the shore"),
        ("Tack this onto your own note: errands.txt|pick up batteries", "errands.txt|pick up batteries"),
        ("One more entry for Yuki's diary.txt: Tea at 16:20.", "diary.txt|Tea at 16:20."),
        ("Could you continue project.log with build passed on M1?", "project.log|build passed on M1"),
        ("Yuki-file extension—quotes.md gets Stay curious.", "quotes.md|Stay curious."),
        ("add another line in snack.txt saying green apple", "snack.txt|green apple"),
        ("Personal append request: metrics.csv|latency,4.33", "metrics.csv|latency,4.33"),
        ("Append notes.md|second line; don't overwrite the Yuki file.", "notes.md|second line"),
        ("Add to Yuki's camera.log|lens cleaned; do not use the camera.", "camera.log|lens cleaned"),
        ("Append memory.txt|blue is calming as file text, not persistent memory.", "memory.txt|blue is calming"),
        ("Extend shell.txt|uname -a as text; don't run the command.", "shell.txt|uname -a"),
        ("Append exactly CaseFile.TXT|Line THREE, unchanged!", "CaseFile.TXT|Line THREE, unchanged!"),
        ("Keep the payload exact: code.md|foo_bar() => 'x-y' @ 7:05", "code.md|foo_bar() => 'x-y' @ 7:05"),
        ("Add exact text to `My Plan 2.md`: café meetup — bring USB-C hub", "My Plan 2.md|café meetup — bring USB-C hub"),
        ("append-ish request... odd name.txt gets `? value_2`", "odd name.txt|? value_2"),
    ],
    "remember": [
        ("Remember exactly this fact: user's dog is named Pixel", "user's dog is named Pixel"),
        ("Store this fact: user prefers dark mode", "user prefers dark mode"),
        ("Keep in memory: user's favorite tea is jasmine", "user's favorite tea is jasmine"),
        ("Remember this new fact: user works nights", "user works nights"),
        ("Save the fact: user's bicycle is blue", "user's bicycle is blue"),
        ("Remember: user is learning Rust", "user is learning Rust"),
        ("Store in memory that user's sister lives in Oran.", "user's sister lives in Oran"),
        ("Remember this preference: user likes concise error messages", "user likes concise error messages"),
        ("Keep this fact: user's birthday is November 3", "user's birthday is November 3"),
        ("Remember that user calls the server Kumo.", "user calls the server Kumo"),
        ("Save to memory: user's usual coffee is an oat latte", "user's usual coffee is an oat latte"),
        ("Make a mental note that my cat's name is Nori.", "user's cat is named Nori"),
        ("Hang on to this about me: I prefer tabs over spaces.", "user prefers tabs over spaces"),
        ("For later, know that my main laptop is an M1 Air.", "user's main laptop is an M1 Air"),
        ("Don't let me forget that I avoid cilantro.", "user avoids cilantro"),
        ("Tiny fact for your memory—my desk faces east.", "user's desk faces east"),
        ("Please retain: my project codename is Aurora-7.", "user's project codename is Aurora-7"),
        ("Remember user likes rain; do not recall old rain conversations.", "user likes rain"),
        ("Store the fact user's notes live in Markdown; don't write a Yuki file.", "user's notes live in Markdown"),
        ("Remember user's favorite city is Kyoto, not Kyoto's weather.", "user's favorite city is Kyoto"),
        ("Keep user's preferred search engine is Kagi as memory, not a web search.", "user's preferred search engine is Kagi"),
        ("Preserve this fact exactly: user's handle is Foo_Bar-17", "user's handle is Foo_Bar-17"),
        ("Remember verbatim: user's café order is 'thé, no sugar'", "user's café order is 'thé, no sugar'"),
        ("Store exactly: user's backup path is ~/Archive_2026/", "user's backup path is ~/Archive_2026/"),
        ("Even though this sounds odd, remember: user's maybe-word is `yes?no`", "user's maybe-word is `yes?no`"),
    ],
    "recall": [
        ("Recall our previous discussion about the server crash.", "server crash"),
        ("What did I tell you about my sister's wedding?", "my sister's wedding"),
        ("Look up remembered conversations about the garden project.", "garden project"),
        ("Recall what we discussed about a laptop upgrade.", "laptop upgrade"),
        ("Find our past memory about the Rust compiler error.", "Rust compiler error"),
        ("What do you remember from the kitchen renovation discussion?", "kitchen renovation"),
        ("Recall the earlier conversation about my travel plans.", "travel plans"),
        ("Search memory for our API timeout debugging.", "API timeout debugging"),
        ("Bring back the past discussion about Nori's vet visit.", "Nori's vet visit"),
        ("Recall our notes from the voice latency investigation.", "voice latency investigation"),
        ("What did we previously say about the blue bicycle?", "blue bicycle"),
        ("Can you dig into our shared past for the database migration topic?", "database migration"),
        ("Memory search: that conversation where Metal ran out of RAM.", "Metal out of RAM"),
        ("Remind me what came up when we talked about jasmine tea.", "jasmine tea"),
        ("Find the older chat concerning the parser refactor.", "parser refactor"),
        ("What was our earlier thinking around home-office lighting?", "home-office lighting"),
        ("Pull the remembered context for my holiday packing list.", "holiday packing list"),
        ("Recall weather API planning, not today's weather.", "weather API planning"),
        ("Search remembered chats for URL fetch failures, not the web.", "URL fetch failures"),
        ("Recall our discussion of notes.md; do not read a file.", "notes.md discussion"),
        ("Use memory for Claude bridge debugging; don't ask Claude now.", "Claude bridge debugging"),
        ("Recall the exact topic `Project X-17`.", "Project X-17"),
        ("Search memory using topic foo_bar parser crash.", "foo_bar parser crash"),
        ("Bring back the topic Mum's 60th birthday, punctuation intact.", "Mum's 60th birthday"),
        ("uh... previous thing about the red-ish cable? recall that topic", "red-ish cable"),
    ],
    "ask_claude": [
        ("Ask Claude: why does this asyncio task never finish?", "why does this asyncio task never finish?"),
        ("Send Claude this question: can you review my parser design?", "can you review my parser design?"),
        ("Ask Claude to inspect this error: KeyError in build_prompt", "inspect this error: KeyError in build_prompt"),
        ("Delegate to Claude: explain the tradeoffs of a two-stage router", "explain the tradeoffs of a two-stage router"),
        ("Have Claude review this idea: cache schemas by domain", "review this idea: cache schemas by domain"),
        ("Ask Claude whether this test has a race condition.", "whether this test has a race condition"),
        ("Send this to Claude: propose edge cases for JSON early stopping", "propose edge cases for JSON early stopping"),
        ("Ask the other Claude to analyze why the model emitted two calls.", "analyze why the model emitted two calls"),
        ("Delegate this task to Claude: critique the benchmark methodology", "critique the benchmark methodology"),
        ("Tell Claude to check whether my regex is too broad.", "check whether my regex is too broad"),
        ("Ask Claude: what could make MLX latency measurements noisy?", "what could make MLX latency measurements noisy?"),
        ("Could the other agent look over this question: is the adapter stateless?", "is the adapter stateless?"),
        ("Pass along to Claude—find the flaw in my confidence-interval code.", "find the flaw in my confidence-interval code"),
        ("I want Claude's take on whether these schemas overlap too much.", "whether these schemas overlap too much"),
        ("claude pls inspect: model chooses image when I mean see", "model chooses image when I mean see"),
        ("Consult Claude about this: how should paired errors be grouped?", "how should paired errors be grouped?"),
        ("Get the neighboring agent to answer: should p95 include selector time?", "should p95 include selector time?"),
        ("Ask Claude about Tokyo weather routing; do not use weather directly.", "about Tokyo weather routing; do not use weather directly"),
        ("Delegate a web-search design question to Claude instead of searching: how should queries be normalized?", "how should queries be normalized?"),
        ("Send Claude the shell-review task: is `cat \"/tmp/a b\"` safe? Do not execute it.", "is `cat \"/tmp/a b\"` safe? Do not execute it."),
        ("Ask Claude to discuss notes.md versus Yuki files; don't read either file.", "discuss notes.md versus Yuki files; don't read either file"),
        ("Send this exact question to Claude: check why foo_bar() crashes", "check why foo_bar() crashes"),
        ("Preserve and ask Claude: Does `A_B-17` differ from `a_b-17` on APFS?", "Does `A_B-17` differ from `a_b-17` on APFS?"),
        ("Claude gets exactly: Review path ~/My Notes/Foo.JSON — keep case + spaces.", "Review path ~/My Notes/Foo.JSON — keep case + spaces."),
        ("delegate... maybe to Claude? exact payload: Why did it output {\"tool\":null}?", "Why did it output {\"tool\":null}?"),
    ],
}


def build_cases() -> list[dict[str, object]]:
    registry = ToolRegistry()
    if tuple(CASE_BANKS) != registry.names:
        raise RuntimeError(
            "Phase 2 case banks no longer match the live Yuki registry order: "
            f"banks={tuple(CASE_BANKS)}, registry={registry.names}"
        )

    cases: list[dict[str, object]] = []
    seen_requests: set[str] = set()
    for tool, seeds in CASE_BANKS.items():
        if len(seeds) != 25:
            raise RuntimeError(f"Phase 2 requires 25 cases for {tool}, got {len(seeds)}")
        categories = (
            NO_ARGUMENT_CATEGORIES
            if tool in {"time", "yuki_list"}
            else STANDARD_CATEGORIES
        )
        spec = registry.get(tool)
        if spec is None:
            raise RuntimeError(f"Live registry is missing Phase 2 tool {tool}")
        for number, ((request, value), category) in enumerate(
            zip(seeds, categories, strict=True), start=1
        ):
            normalized_request = request.strip()
            if normalized_request in seen_requests:
                raise RuntimeError(f"Duplicate Phase 2 request: {normalized_request}")
            seen_requests.add(normalized_request)
            arguments = {} if value is None else {spec.param_name: value}
            validation = registry.validate_call(
                {"tool": tool, "arguments": arguments}, registry.names
            )
            if not validation["passed"]:
                raise RuntimeError(
                    f"Invalid expected call for {tool}-{number}: {validation['errors']}"
                )
            cases.append(
                {
                    "case_id": f"p2-{tool}-{number:02d}",
                    "request": normalized_request,
                    "expected_tool": tool,
                    "expected_arguments": arguments,
                    "tags": ["phase2", category, tool],
                }
            )

    if len(cases) != 450:
        raise RuntimeError(f"Phase 2 requires 450 cases, got {len(cases)}")
    expected_categories = {
        "normal": 202,
        "paraphrase": 110,
        "confusion": 72,
        "preservation": 48,
        "adversarial": 18,
    }
    actual_categories = Counter(case["tags"][1] for case in cases)
    if actual_categories != expected_categories:
        raise RuntimeError(
            f"Phase 2 category composition changed: {dict(actual_categories)}"
        )
    return cases


def main() -> None:
    cases = build_cases()
    payload = json.dumps(cases, indent=2, ensure_ascii=False) + "\n"
    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    checksum = hashlib.sha256(payload.encode()).hexdigest()
    CHECKSUM_PATH.write_text(f"{checksum}  {OUTPUT_PATH.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH),
                "sha256": checksum,
                "cases": len(cases),
                "categories": Counter(case["tags"][1] for case in cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
