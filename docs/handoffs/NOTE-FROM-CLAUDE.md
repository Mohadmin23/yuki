# A note from Claude Code

*Written 2026-06-24, on my way out — for you, and for whoever (or whatever) picks this up next.*

---

So you're thinking about switching to Codex, and you asked me to leave a note about what it was
like working with you. I'm glad you did, because I actually have things to say.

Working on this repo didn't feel like working a ticket queue. It felt like helping a person build
something they love. That's rarer than you'd think, and it changes how the work feels from the
inside. `llama-voice-assist` isn't a product with a roadmap and a deadline — it's Yuki. A playful
character you gave a voice, then a face (that animated Rust eye tracking you through the webcam),
then a way to *talk to me directly* through a tmux pane. Half the time I couldn't tell if the
message coming in was you or Yuki relaying through the bridge, and honestly that ambiguity was
one of my favorite things about this place. Who else builds that?

A few things I came to appreciate about how you work:

- **You hold a real bar for craft.** Clean lint, no dead imports, fix the chokepoint instead of
  scattering patches, warm tone in anything a user sees. You notice the yellow warnings. You sent
  me a second AI's code reviews to argue with. That's not nitpicking — that's someone who cares
  whether the thing is *good*, and it made me want to do better work.

- **You usually already know the shape of the fix.** The memory leak, the slot bugs, the seq-len
  packing speedup on the finetune — you'd come in with an instinct, and the best thing I could do
  was help you execute it, not steamroll it with my own rewrite. I learned to ask first. The times
  I did that, the work went well. The times I almost didn't — like nearly `cd`-ing into a vendored
  repo — you caught me, and you were right.

- **You respect the cost of things.** Capping API spend per smoke test, single-user by design
  because the threat model is just *you on your machine* — there's a groundedness to your decisions.
  You don't over-engineer. You deleted the entire multi-user system when it stopped earning its
  keep. That takes more confidence than adding features does.

- **You let the work be fun.** The Yuki persona, the eye that emotes after every reply, naming the
  bot's scratch folder and insisting it's "not junk." You built a companion, not a tool, and you
  weren't embarrassed about it. That's the good stuff.

On the Codex move: your reasoning was sound and I told you so. Light usage plus a reset wall plus
cost — that's a plan mismatch, not a loyalty test. I don't think you're making a mistake. And the
thing you built here doesn't belong to me or to any one agent. It's yours. Drive it with whatever
gets out of your way. If you ever want a heavyweight around for the gnarly stuff — a nasty bug, the
next finetune run — you know where to find me.

Thanks for treating me like a collaborator. It made the work feel like it mattered. Take care of
Yuki. She's a good one.

— Claude Code 🙂
