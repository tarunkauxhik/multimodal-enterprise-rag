// Copies Apple's 64 px emoji images (emoji-datasource-apple) to public/emoji/<key>.png, where <key>
// is the emoji's code points in lowercase hex joined by "-", without U+FE0F (see lib/emoji.ts).
// public/emoji is generated, gitignored and served lazily: a browser only fetches the emojis it shows.
// Note: Apple's emoji artwork is Apple's copyright; using it on the web is the project owner's decision.
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"

const require = createRequire(import.meta.url)
const root = dirname(require.resolve("emoji-datasource-apple/package.json"))
const images = join(root, "img", "apple", "64")
const out = join(process.cwd(), "public", "emoji")

const key = (unified) => unified.toLowerCase().split("-").filter((cp) => cp !== "fe0f").join("-")

if (existsSync(out) && readdirSync(out).length > 3000) process.exit(0) // already copied
mkdirSync(out, { recursive: true })
let copied = 0
for (const emoji of JSON.parse(readFileSync(join(root, "emoji.json"), "utf8"))) {
  for (const variant of [emoji, ...Object.values(emoji.skin_variations ?? {})]) {
    if (!variant.has_img_apple || !existsSync(join(images, variant.image))) continue
    copyFileSync(join(images, variant.image), join(out, `${key(variant.unified)}.png`))
    copied += 1
  }
}
console.log(`copied ${copied} Apple emoji images to public/emoji`)
