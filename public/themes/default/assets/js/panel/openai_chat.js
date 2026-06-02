// cspell:disable

/**
 * @typedef {Object} AiResponse
 * @property {{slug: string, label: string}} model - The AI model information
 * @property {HTMLElement} bubbleEl - The bubble element
 * @property {HTMLElement} chatContentEl - The chat content element
 * @property {HTMLElement} chatContentContainerEl - The chat content container element
 * @property {HTMLElement | null} acceptButtonEl - The accept button element
 * @property {HTMLElement | null} regenerateButtonEl - The regenrate button element
 * @property {boolean} responseStreaming - Whether response is streaming
 * @property {number} responseId - Id of the message coming from backend
 * @property {string[]} response - Array of response strings
 * @property {AbortController | null} abortController - Abort controller
 * @property {HTMLElement[] | null} placeholderEls - Placeholder elements for images
 * @property {Object | null} request - The request object
 * @property {number} animatingWordIndex - The index of animating word
 * @property {Set<HTMLElement>} animatedElements - Set of already animated elements
 * @property {number} lastAnimatedElOffsetTop - Offset top of the last animated element
 */

/**
 * @type {AiResponse[]}
 */
let aiResponses = [];
let selectedPrompt = -1;
let promptsData = [];
let favData = [];
let searchString = '';
let pdf = undefined;
let pdfName = '';
let pdfPath = '';
let filterType = 'all';
let chatAttachments = [];
let navigatingInChatsHistory = false;
let selectedHistoryPrompt = -1;
let councilChildAbortControllers = [];
let councilRunSequence = 0;

/**
 * Credits: Joydeep Bhowmik https://dev.to/joydeep23/adding-keys-our-dom-diffing-algorithm-4d7g
 */
class LiquidVDOM {
	getnodeType(node) {
		if (node.nodeType == 1) return node.tagName.toLowerCase();
		else return node.nodeType;
	}

	clean(node) {
		for (let n = 0; n < node.childNodes.length; n++) {
			let child = node.childNodes[n];
			if (child.nodeType === 8) {
				// Only remove comment nodes
				node.removeChild(child);
				n--;
			} else if (child.nodeType === 1) {
				// Element node
				if (child.hasAttribute('key')) {
					let key = child.getAttribute('key');
					child.key = key;
					child.removeAttribute('key');
				}
				this.clean(child);
			}
		}
	}

	parseHTML(str) {
		let parser = new DOMParser();
		let doc = parser.parseFromString(str, 'text/html');
		this.clean(doc.body);
		return doc.body;
	}

	attrbutesIndex(el) {
		var attributes = {};
		if (el.attributes == undefined) return attributes;
		for (var i = 0, atts = el.attributes, n = atts.length; i < n; i++) {
			attributes[atts[i].name] = atts[i].value;
		}
		return attributes;
	}

	patchAttributes(vdom, dom) {
		let vdomAttributes = this.attrbutesIndex(vdom);
		let domAttributes = this.attrbutesIndex(dom);
		if (vdomAttributes == domAttributes) return;
		Object.keys(vdomAttributes).forEach((key, i) => {
			//if the attribute is not present in dom then add it
			if (!dom.getAttribute(key)) {
				dom.setAttribute(key, vdomAttributes[key]);
			} //if the atrtribute is present than compare it
			else if (dom.getAttribute(key)) {
				if (vdomAttributes[key] != domAttributes[key]) {
					dom.setAttribute(key, vdomAttributes[key]);
				}
			}
		});
		Object.keys(domAttributes).forEach((key, i) => {
			//if the attribute is not present in vdom than remove it
			if (!vdom.getAttribute(key)) {
				dom.removeAttribute(key);
			}
		});
	}

	hasTheKey(dom, key) {
		let keymatched = false;
		for (let i = 0; i < dom.children.length; i++) {
			if (key == dom.children[i].key) {
				keymatched = true;
				break;
			}
		}
		return keymatched;
	}

	patchKeys(vdom, dom) {
		//remove unmatched keys from dom
		for (let i = 0; i < dom.children.length; i++) {
			let dnode = dom.children[i];
			let key = dnode.key;
			if (key) {
				if (!this.hasTheKey(vdom, key)) {
					dnode.remove();
				}
			}
		}
		//adding keys to dom
		for (let i = 0; i < vdom.children.length; i++) {
			let vnode = vdom.children[i];
			let key = vnode.key;
			if (key) {
				if (!this.hasTheKey(dom, key)) {
					//if key is not present in dom then add it
					let nthIndex = [].indexOf.call(
						vnode.parentNode.children,
						vnode,
					);
					if (dom.children[nthIndex]) {
						dom.children[nthIndex].before(vnode.cloneNode(true));
					} else {
						dom.append(vnode.cloneNode(true));
					}
				}
			}
		}
	}

	diff(vdom, dom) {
		//if dom has no childs then append the childs from vdom
		if (dom.hasChildNodes() == false && vdom.hasChildNodes() == true) {
			for (let i = 0; i < vdom.childNodes.length; i++) {
				//appending
				dom.append(vdom.childNodes[i].cloneNode(true));
			}
		} else {
			this.patchKeys(vdom, dom);
			//if dom has extra child
			if (dom.childNodes.length > vdom.childNodes.length) {
				let count = dom.childNodes.length - vdom.childNodes.length;
				if (count > 0) {
					for (; count > 0; count--) {
						dom.childNodes[dom.childNodes.length - count].remove();
					}
				}
			}
			//now comparing all childs
			for (let i = 0; i < vdom.childNodes.length; i++) {
				//if the node is not present in dom append it
				if (dom.childNodes[i] == undefined) {
					dom.append(vdom.childNodes[i].cloneNode(true));
					// console.log("appenidng",vdom.childNodes[i])
				} else if (
					this.getnodeType(vdom.childNodes[i]) ==
					this.getnodeType(dom.childNodes[i])
				) {
					//if same node type
					//if the nodeType is text
					if (vdom.childNodes[i].nodeType == 3) {
						//we check if the text content is not same
						if (
							vdom.childNodes[i].textContent !=
							dom.childNodes[i].textContent
						) {
							//replace the text content
							dom.childNodes[i].textContent =
								vdom.childNodes[i].textContent;
						}
					} else {
						this.patchAttributes(
							vdom.childNodes[i],
							dom.childNodes[i],
						);
					}
				} else {
					//replace
					dom.childNodes[i].replaceWith(
						vdom.childNodes[i].cloneNode(true),
					);
				}
				if (vdom.childNodes[i].nodeType != 3) {
					// Skip diffing into Alpine-managed elements — their children
					// are rendered by Alpine and must not be touched by VDOM.
					if (dom.childNodes[i]._x_dataStack) continue;

					this.diff(vdom.childNodes[i], dom.childNodes[i]);
				}
			}
		}
	}
}

const liquidVDOM = new LiquidVDOM();

function unwrapWords(node) {
	if (
		node.nodeName === 'PRE' ||
		node.nodeName === 'CODE' ||
		node.nodeName === 'A' ||
		node.nodeName === 'TR' ||
		node.classList?.contains('katex')
	) return;

	if (node.classList?.contains('done-signal')) {
		return node.remove();
	}

	if (node.nodeType === 3) {
		return;
	}

	if (node.nodeName === 'SPAN' && node.classList?.contains('animated-el')) {
		const textNode = document.createTextNode(node.textContent);
		node.parentNode.replaceChild(textNode, node);
		return;
	}

	const childNodes = [ ...node.childNodes ];
	childNodes.forEach(child => unwrapWords(child));
}

function generateUUID() {
	return ([ 1e7 ] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
		(c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
	);
}

function buildPrompt(string) {
	const chatsV2 = Alpine.store('chatsV2');
	let prompt = string.trim();

	if ( prompt && chatsV2 && chatsV2.selectedTools ) {
		const followUps = chatsV2.selectedTools.filter(tool => tool.startsWith('entity-follow-up-'));

		if ( followUps.length ) {
			const existingPrompt = prompt || '';
			prompt = `${magicai_localize.regarding ?? 'Regarding'} ${followUps.map(f => f.replace('entity-follow-up-', '')).join(', ')}: ${existingPrompt}`;
		}
	}

	return prompt;
}

/**
 * @param {Object} param0
 * @param {AiResponse} param0.responseObj
 * @param {boolean} param0.withoutDone
 */
function getAiResponseString({ responseObj = null, withoutDone = true }) {
	if (!responseObj) {
		responseObj = aiResponses[0];
	}

	if (!responseObj) return '';

	let string = responseObj.response
		.join('')
		.trim()
		.replace(/<br\s*\/?>/g, '\n');

	if (withoutDone) {
		return string.replace('[DONE]', '');
	}

	return string;
}

function fixUnclosedMarkdownSyntax(string) {
	let text = string;

	let boldMatch = text.match(/\*\*(?:(?!\*\*).)*$/);
	if (boldMatch) {
		text = text + '**';
	}

	let italicMatch = text.match(/\*(?:(?!\*).)*$/);
	if (italicMatch) {
		text = text + '*';
	}

	let codeBlockMatch = text.match(/```(?:(?!```).)*$/);
	if (codeBlockMatch) {
		text = text + '```';
	}

	let inlineCodeMatch = text.match(/`(?:(?!`).)*$/);
	if (inlineCodeMatch) {
		text = text + '`';
	}

	let strikeMatch = text.match(/~~(?:(?!~~).)*$/);
	if (strikeMatch) {
		text = text + '~~';
	}

	return text;
}

function parseSkillFrontmatter(raw) {
	const match = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/);
	if (!match) return { name: 'Untitled Skill', description: '', body: raw.trim(), raw: raw };
	const frontmatter = match[1];
	const body = match[2].trim();
	let name = 'Untitled Skill';
	let description = '';
	const nameMatch = frontmatter.match(/name:\s*"?([^"\n]+)"?/);
	if (nameMatch) name = nameMatch[1].trim();
	const descMatch = frontmatter.match(/description:\s*"?([^"\n]+)"?/);
	if (descMatch) description = descMatch[1].trim();
	return { name, description, body, raw: raw };
}

/**
 * Parse multi-file skill format using ===FILE: path=== delimiters.
 * Falls back to single SKILL.md if no delimiters found.
 */
function parseMultiFileSkill(str) {
	const delimiter = /^===FILE:\s*(.+?)\s*===$/gm;
	const parts = [];
	let lastIndex = 0;
	let lastPath = null;
	let match;

	while ((match = delimiter.exec(str)) !== null) {
		if (lastPath !== null) {
			parts.push({ path: lastPath, content: str.substring(lastIndex, match.index).trim() });
		}
		lastPath = match[1];
		lastIndex = match.index + match[0].length;
	}

	if (lastPath !== null) {
		parts.push({ path: lastPath, content: str.substring(lastIndex).trim() });
	}

	// No delimiters found — treat entire content as SKILL.md
	if (parts.length === 0) {
		return { files: [ { path: 'SKILL.md', content: str.trim() } ] };
	}

	return { files: parts };
}

/**
 * Build bundled_resources object from parsed files for the file tree.
 */
function buildBundledResourcesFromFiles(files) {
	const resources = {};
	for (const file of files) {
		if (file.path.toUpperCase() === 'SKILL.MD') continue;
		const parts = file.path.split('/');
		if (parts.length >= 2) {
			const folder = parts[0];
			if (!resources[folder]) resources[folder] = [];
			resources[folder].push({ name: parts.slice(1).join('/'), path: file.path, size: file.content.length });
		}
	}
	return Object.keys(resources).length > 0 ? resources : null;
}

window.downloadSkillFile = function(encodedContent, fileName) {
	var decoded = decodeURIComponent(escape(atob(encodedContent)));
	var blob = new Blob([ decoded ], { type: 'text/markdown' });
	var url = window.URL.createObjectURL(blob);
	var a = document.createElement('a');
	a.href = url;
	a.download = fileName;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	window.URL.revokeObjectURL(url);
};

/**
 * Download a skill as a .skill file (zip format with .skill extension).
 * Compatible with Claude, OpenAI, and other skill platforms.
 * Uses a minimal zip builder — no external dependencies.
 */
window.downloadSkillZip = function(encodedContent) {
	var decoded = decodeURIComponent(escape(atob(encodedContent)));
	var parsed = parseMultiFileSkill(decoded);
	var skillMd = parsed.files.find(function(f) { return f.path.toUpperCase() === 'SKILL.MD'; });
	var meta = skillMd ? parseSkillFrontmatter(skillMd.content) : { name: 'skill' };
	var slugName = (meta.name || 'skill').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-+$/, '');

	// Build zip file entries
	var entries = [];
	parsed.files.forEach(function(file) {
		entries.push({ path: file.path, data: new TextEncoder().encode(file.content) });
	});

	var zipBlob = buildZipBlob(entries);
	var url = window.URL.createObjectURL(zipBlob);
	var a = document.createElement('a');
	a.href = url;
	a.download = slugName + '.skill';
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	window.URL.revokeObjectURL(url);
};

/**
 * Build a zip Blob from an array of {path, data (Uint8Array)} entries.
 * Minimal zip implementation — store method (no compression), compatible with all unzip tools.
 */
function buildZipBlob(entries) {
	var localHeaders = [];
	var centralHeaders = [];
	var offset = 0;

	entries.forEach(function(entry) {
		var pathBytes = new TextEncoder().encode(entry.path);
		var data = entry.data;
		var crc = crc32(data);

		// Local file header (30 bytes + path + data)
		var local = new Uint8Array(30 + pathBytes.length + data.length);
		var v = new DataView(local.buffer);
		v.setUint32(0, 0x04034b50, true);  // signature
		v.setUint16(4, 20, true);           // version needed
		v.setUint16(6, 0, true);            // flags
		v.setUint16(8, 0, true);            // compression: store
		v.setUint16(10, 0, true);           // mod time
		v.setUint16(12, 0, true);           // mod date
		v.setUint32(14, crc, true);         // crc-32
		v.setUint32(18, data.length, true); // compressed size
		v.setUint32(22, data.length, true); // uncompressed size
		v.setUint16(26, pathBytes.length, true); // filename length
		v.setUint16(28, 0, true);           // extra field length
		local.set(pathBytes, 30);
		local.set(data, 30 + pathBytes.length);
		localHeaders.push(local);

		// Central directory header (46 bytes + path)
		var central = new Uint8Array(46 + pathBytes.length);
		var cv = new DataView(central.buffer);
		cv.setUint32(0, 0x02014b50, true);  // signature
		cv.setUint16(4, 20, true);           // version made by
		cv.setUint16(6, 20, true);           // version needed
		cv.setUint16(8, 0, true);            // flags
		cv.setUint16(10, 0, true);           // compression
		cv.setUint16(12, 0, true);           // mod time
		cv.setUint16(14, 0, true);           // mod date
		cv.setUint32(16, crc, true);         // crc-32
		cv.setUint32(20, data.length, true); // compressed size
		cv.setUint32(24, data.length, true); // uncompressed size
		cv.setUint16(28, pathBytes.length, true); // filename length
		cv.setUint16(30, 0, true);           // extra field length
		cv.setUint16(32, 0, true);           // comment length
		cv.setUint16(34, 0, true);           // disk number start
		cv.setUint16(36, 0, true);           // internal file attributes
		cv.setUint32(38, 0, true);           // external file attributes
		cv.setUint32(42, offset, true);      // local header offset
		central.set(pathBytes, 46);
		centralHeaders.push(central);

		offset += local.length;
	});

	// End of central directory
	var centralSize = centralHeaders.reduce(function(sum, h) { return sum + h.length; }, 0);
	var eocd = new Uint8Array(22);
	var ev = new DataView(eocd.buffer);
	ev.setUint32(0, 0x06054b50, true);          // signature
	ev.setUint16(4, 0, true);                    // disk number
	ev.setUint16(6, 0, true);                    // central dir disk
	ev.setUint16(8, entries.length, true);        // entries on disk
	ev.setUint16(10, entries.length, true);       // total entries
	ev.setUint32(12, centralSize, true);          // central dir size
	ev.setUint32(16, offset, true);               // central dir offset
	ev.setUint16(20, 0, true);                    // comment length

	var parts = localHeaders.concat(centralHeaders, [ eocd ]);
	return new Blob(parts, { type: 'application/zip' });
}

/**
 * CRC-32 computation for zip file integrity.
 */
function crc32(data) {
	var table = crc32.table;
	if (!table) {
		table = crc32.table = new Uint32Array(256);
		for (var i = 0; i < 256; i++) {
			var c = i;
			for (var j = 0; j < 8; j++) {
				c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
			}
			table[i] = c;
		}
	}
	var crc = 0xFFFFFFFF;
	for (var k = 0; k < data.length; k++) {
		crc = table[(crc ^ data[k]) & 0xFF] ^ (crc >>> 8);
	}
	return (crc ^ 0xFFFFFFFF) >>> 0;
}

/**
 * Add a skill to the user's collection from chat output.
 */
window.addSkillFromContent = function(encodedContent, buttonEl) {
	var decoded = decodeURIComponent(escape(atob(encodedContent)));
	var parsed = parseMultiFileSkill(decoded);

	if (buttonEl) {
		buttonEl.disabled = true;
		buttonEl.textContent = 'Adding...';
	}

	fetch('/dashboard/user/skills/create-from-content', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '',
			'X-Requested-With': 'XMLHttpRequest',
		},
		body: JSON.stringify({ files: parsed.files }),
	})
		.then(function(res) { return res.json(); })
		.then(function(data) {
			if (data.message) {
				toastr.success(data.message);
			}
			if (buttonEl) {
				buttonEl.textContent = 'Added';
				buttonEl.classList.add('opacity-50', 'pointer-events-none');
			}
		})
		.catch(function() {
			toastr.error('Failed to add skill.');
			if (buttonEl) {
				buttonEl.disabled = false;
				buttonEl.textContent = 'Add to My Skills';
			}
		});
};

/**
 * Open the skill preview modal from a chat-generated skill.
 */
window.openSkillPreview = function(encodedContent) {
	var decoded = decodeURIComponent(escape(atob(encodedContent)));
	var parsed = parseMultiFileSkill(decoded);
	var skillMd = parsed.files.find(function(f) { return f.path.toUpperCase() === 'SKILL.MD'; });
	var meta = skillMd ? parseSkillFrontmatter(skillMd.content) : { name: 'Untitled Skill', description: '', body: '' };
	var bundledResources = buildBundledResourcesFromFiles(parsed.files);

	window.dispatchEvent(new CustomEvent('open-skill-preview', {
		detail: {
			name: meta.name,
			description: meta.description,
			instructions: meta.body,
			bundled_resources: bundledResources,
			files: parsed.files,
		},
	}));
};

function renderSkillCard(str) {
	var parsed = parseMultiFileSkill(str);
	var skillMd = parsed.files.find(function(f) { return f.path.toUpperCase() === 'SKILL.MD'; });
	var meta = skillMd ? parseSkillFrontmatter(skillMd.content) : { name: '', description: '' };
	var rawEncoded = btoa(unescape(encodeURIComponent(str)));
	var fileCount = parsed.files.length;
	var safeName = escapeHtml(meta.name || 'Untitled Skill');
	var safeDesc = escapeHtml(meta.description || '');

	// markdownit: if highlight returns a string starting with <pre, it won't add its own <pre><code> wrapper.
	// We use a <pre> with display:contents to be a transparent container that avoids double-wrapping.

	// Check if this is a partial/streaming skill (no valid name yet)
	if (!meta.name || meta.name === 'Untitled Skill') {
		// Shimmer loading card
		return '<pre style="display:contents;all:unset">' +
		'<div class="lqd-skill-card-shimmer rounded-xl border border-card-border bg-card-background p-5 my-2">' +
			'<div class="flex items-center gap-2 mb-3">' +
				'<div class="size-8 rounded-lg bg-foreground/5 animate-pulse"></div>' +
				'<div class="flex-1">' +
					'<div class="h-4 w-40 rounded bg-foreground/5 animate-pulse mb-1.5"></div>' +
					'<div class="h-3 w-56 rounded bg-foreground/5 animate-pulse"></div>' +
				'</div>' +
			'</div>' +
			'<div class="flex items-center gap-2">' +
				'<div class="h-8 w-24 rounded-lg bg-foreground/5 animate-pulse"></div>' +
				'<div class="h-8 w-28 rounded-lg bg-foreground/5 animate-pulse"></div>' +
				'<div class="h-8 w-32 rounded-lg bg-foreground/5 animate-pulse"></div>' +
			'</div>' +
			'<p class="mt-2 text-2xs lqd-shimmer-text bg-clip-text bg-gradient-to-r from-foreground via-background to-foreground text-transparent">Creating skill...</p>' +
		'</div>' +
		'</pre>';
	}

	var editIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="size-4"><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path d="M4 20h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4" /><path d="M13.5 6.5l4 4" /></svg>';
	var linkIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor" class="size-5"><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path d="M15.707 8.293a1 1 0 0 1 0 1.414l-6 6a1 1 0 1 1 -1.414 -1.414l6 -6a1 1 0 0 1 1.414 0" /><path d="M19.242 4.757c2.343 2.344 2.342 6.143 -.052 8.534l-.534 .464a1 1 0 1 1 -1.312 -1.51l.483 -.416a4 4 0 0 0 0 -5.657c-1.562 -1.563 -4.095 -1.563 -5.607 -.054l-.463 .536a1 1 0 1 1 -1.514 -1.308l.513 -.59a6 6 0 0 1 8.486 .001" /><path d="M6.75 10.338a1 1 0 0 1 -.088 1.411l-.483 .425a3.97 3.97 0 0 0 0 5.649a4.064 4.064 0 0 0 5.678 .038l.34 -.458a1 1 0 1 1 1.606 1.194l-.397 .534l-.1 .114a6.07 6.07 0 0 1 -8.533 0a5.97 5.97 0 0 1 -1.773 -4.247c0 -1.595 .638 -3.124 1.814 -4.284l.524 -.463a1 1 0 0 1 1.411 .087" /></svg>';

	return '<pre style="display:contents;all:unset">' +
	'<div class="lqd-skill-card rounded-[20px] bg-card-background p-4 shadow-[0_4px_44px_hsl(0_0%_0%/5%)] md:p-5">' +
		'<button type="button" class="group mb-3 inline-flex items-center gap-1.5 text-2xs font-medium text-foreground/70 transition-colors hover:text-foreground" onclick="window.openSkillPreview(\'' + rawEncoded + '\')">' +
			'<span class="inline-grid size-7 place-items-center rounded-full border border-foreground/15 shadow-xs shadow-black/5 transition group-hover:scale-105">' + editIcon + '</span>' +
			' ' + (magicai_localize.open_skill ?? 'Open Skill') + '</button>' +
		'<h4 class="mt-0 mb-2.5 text-2xs leading-tight font-medium">' + safeName + '</h4>' +
		(safeDesc ? '<p class="mt-0 mb-2.5">' + safeDesc + '</p>' : '') +
		(fileCount > 1 ? '<span class="mt-1.5 inline-flex items-center rounded-full bg-foreground/5 px-2 py-0.5 text-3xs text-foreground/50">' + fileCount + ' files</span>' : '') +
		'<div class="mt-3">' +
			'<a href="#" class="inline-flex items-center gap-2 text-xs font-medium text-primary underline decoration-primary/10 underline-offset-4 transition-colors hover:decoration-primary" onclick="event.preventDefault();window.downloadSkillZip(\'' + rawEncoded + '\')">' + linkIcon + ' Download Skill</a>' +
		'</div>' +
	'</div>' +
	'</pre>';
}

/**
 * @param {string} string
 * @param {object} options
 * @param {boolean} options.readyForAnimation
 */
function formatString(string, options = {}) {
	if (!('markdownit' in window)) return;

	string = fixUnclosedMarkdownSyntax(string);

	string = string
		.replace(
			/(?<=\[START_REASONING\])(?:.*?\n\n.*?)(?=\[END_REASONING\]|$)/gs,
			match => match.replace(/\n\n/g, '\n'),
		)
		.replace('[START_REASONING]', '>')
		.replace('[END_REASONING]', '\n')
		.replaceAll('\\(', '$')
		.replaceAll('\\)', '$')
		.replaceAll('\\[', '$$')
		.replaceAll('\\]', '$$');

	const renderer = window.markdownit({
		breaks: true,
		highlight: (str, lang) => {
			const language = lang && lang !== '' ? lang : 'md';

			if (language === 'skill') {
				return renderSkillCard(str);
			}

			const codeString = str;

			const highlighted = Prism.highlight(
				codeString,
				(Prism.languages[language] != null ? Prism.languages[language] : (language === 'blade' ? Prism.languages.html : Prism.languages.markup)),
				language,
			);

			return `<pre class="${options.readyForAnimation ? 'animated-el' : ''} !whitespace-pre-wrap rounded [direction:ltr] max-w-full !w-full language-${language}"><code data-lang="${language}" class="language-${language}">${highlighted}</code></pre>`;
		},
	});

	if ('katex' in window && 'markdownItKatex' in window) {
		renderer.use(markdownItKatex);
	}

	if ('markdownitContainer' in window) {
		const containers = [
			'social-media-agent-chat-post-card',
			'social-media-agent-chat-post-card-head',
			'social-media-agent-chat-post-card-platform',
			'social-media-agent-chat-post-card-info',
			'social-media-agent-chat-post-card-images',
			'social-media-agent-chat-post-card-content',
			'social-media-agent-chat-post-card-foot',
			'lqd-chat-image-grid',
			'smart-images',
			'entity-highlights',
			'meta',
		];

		containers.forEach(container => {
			let options = {};

			if ( container === 'smart-images' ) {
				options = {
					render: function (tokens, idx) {
						if (tokens[idx].nesting === 1) {
							return '<div class="lqd-smart-images-container hidden">\n';
						} else {
							return '</div>\n';
						}
					}
				};
			}

			if ( container === 'entity-highlights' || container === 'meta' ) {
				options = {
					render: function (tokens, idx) {
						if (tokens[idx].nesting === 1) {
							return '<div class="lqd-entity-highlights-data hidden" style="display:none">\n';
						} else {
							return '</div>\n';
						}
					}
				};
			}

			if ( container === 'social-media-agent-chat-post-card' ) {
				options = {
					render: function (tokens, idx) {
						if (tokens[idx].nesting === 1) {
							const token = tokens[idx];

							let attributes = 'class="social-media-agent-chat-post-card" x-data="socialMediaAgentChatPostCard" @social-media-agent-post-updated.window="onPostUpdated" @social-media-agent-post-rejected.window="onPostRejected"';

							if ( token.attrs && token.attrs.length ) {
								token.attrs.forEach(([ key, val ]) => attributes += ` ${key}="${val}"`);
							}

							return '<div ' + attributes + '>\n';
						} else {
							// closing tag
							return '</div>\n';
						}
					}
				};
			}

			renderer.use(markdownitContainer, container, options);
		});
	}

	renderer.use(function (md) {
		md.core.ruler.push('filter-lqd-chat-image-grid', function (state) {
			let inGrid = false;
			const toRemove = new Set();

			for (let i = 0; i < state.tokens.length; i++) {
				const token = state.tokens[i];

				if (token.type === 'container_lqd-chat-image-grid_open') {
					inGrid = true;
					continue;
				}

				if (token.type === 'container_lqd-chat-image-grid_close') {
					inGrid = false;
					continue;
				}

				if (!inGrid) continue;

				if (token.type === 'inline' && token.children) {
					let insideLink = false;

					token.children = token.children.filter(function (child) {
						if (child.type === 'link_open') { insideLink = true; return true; }
						if (child.type === 'link_close') { insideLink = false; return true; }
						if (child.type === 'image') return true;
						return insideLink;
					});

					if (token.children.length === 0) {
						toRemove.add(i);
						if (i > 0 && state.tokens[i - 1].type.endsWith('_open')) toRemove.add(i - 1);
						if (i < state.tokens.length - 1 && state.tokens[i + 1].type.endsWith('_close')) toRemove.add(i + 1);
					}
				}
			}

			if (toRemove.size > 0) {
				state.tokens = state.tokens.filter(function (_, idx) { return !toRemove.has(idx); });
			}
		});
	});

	if ('markdownItAttrs' in window) {
		renderer.use(markdownItAttrs);
	}

	renderer.use(function (md) {
		// Add data-fslightbox attribute to images
		const defaultRender = md.renderer.rules.image;

		md.renderer.rules.image = function (tokens, idx, options, env, self) {
			const token = tokens[idx];
			// Find the src attribute to use as the href
			const srcIndex = token.attrIndex('src');
			const src = srcIndex >= 0 ? token.attrs[srcIndex][1] : '';

			// Render the image with default renderer
			const imageHtml = defaultRender(tokens, idx, options, env, self);

			// Wrap in anchor tag with href to the image source
			return `<a href="${src}" target="_blank" data-fslightbox="gallery">${imageHtml}</a>`;
		};

		// Detect and wrap HTML code blocks
		md.core.ruler.before('block', 'detect_html_blocks', function (state) {
			const src = state.src;

			// Look for HTML that starts with <html> or <!DOCTYPE html> and isn't already in code blocks
			const htmlStartRegex = /(?:^|\n)(?!```)(<!DOCTYPE\s+html[^>]*>[\s\S]*?<html[^>]*>|<html[^>]*>)/gi;
			let match;
			let modifiedSrc = src;
			let offset = 0;

			while ((match = htmlStartRegex.exec(src)) !== null) {
				const startPos = match.index + offset;
				const htmlStart = match[0];

				// Check if this HTML start is already inside a code block
				const beforeContent = modifiedSrc.substring(0, startPos);
				const codeBlockCount = (beforeContent.match(/```/g) || []).length;
				const isInsideCodeBlock = codeBlockCount % 2 === 1;

				if (!isInsideCodeBlock) {
					// Check if the HTML content doesn't already have ``` at the beginning
					const startsWithCodeBlock = htmlStart.trim().startsWith('```');

					if (!startsWithCodeBlock) {
						// Ensure the code block starts on a new line
						const precedingChar = startPos > 0 ? modifiedSrc.charAt(startPos - 1) : '\n';
						const needsNewline = precedingChar !== '\n';

						// Start wrapping immediately with opening code block
						const codeBlockStart = (needsNewline ? '\n' : '') + '```html\n';

						// Insert the opening code block
						modifiedSrc = modifiedSrc.substring(0, startPos) + codeBlockStart + modifiedSrc.substring(startPos);

						// Update offset
						offset += codeBlockStart.length;

						// Now look for the closing </html> tag in the modified content
						const afterStart = modifiedSrc.substring(startPos + codeBlockStart.length);
						const htmlEndMatch = afterStart.match(/<\/html\s*>/i);

						if (htmlEndMatch) {
							const endPos = startPos + codeBlockStart.length + htmlEndMatch.index + htmlEndMatch[0].length;

							// Check if there's already a newline after </html>
							const nextChar = endPos < modifiedSrc.length ? modifiedSrc.charAt(endPos) : '';
							const hasFollowingNewline = nextChar === '\n';

							// Insert closing code block after </html> with proper newlines
							const codeBlockEnd = '\n```' + (hasFollowingNewline ? '' : '\n');
							modifiedSrc = modifiedSrc.substring(0, endPos) + codeBlockEnd + modifiedSrc.substring(endPos);

							// Update offset for next iterations
							offset += codeBlockEnd.length;
						}
					}
				}
			}

			if (modifiedSrc !== src) {
				state.src = modifiedSrc;
			}
		});

		// Wrap words with animated-el spans for animation
		if (options.readyForAnimation) {
			md.core.ruler.after('inline', 'wrap_words', function (state) {
				state.tokens.forEach(function (blockToken) {
					if (blockToken.type !== 'inline') return;

					const inlineElements = [ 'strong', 'em', 's', 'u', 'a', 'i', 'b', 'code', 'del', 'ins', 'mark', 'sub', 'sup' ];
					let insideInlineElement = false;

					blockToken.children.forEach(function (token) {
						if (token.type === 'text' && !insideInlineElement) {
							// Split text into words and wrap each with span
							const words = token.content.split(/(\s+)/);
							let wrappedContent = '';

							words.forEach(word => {
								if (word.trim() !== '') {
									wrappedContent += `<span class="animated-el ${word.includes('[DONE]') ? 'done-signal' : ''}">${word}</span>`;
								} else {
									wrappedContent += word; // Preserve whitespace
								}
							});

							// Convert to HTML token
							token.type = 'html_inline';
							token.content = wrappedContent;
						}

						// Track if we're inside inline elements and add animated-el class
						if (token.type.endsWith('_open')) {
							const tagName = token.tag;
							if (inlineElements.includes(tagName)) {
								insideInlineElement = true;
								// Add animated-el class to inline elements
								if (!token.attrGet || !token.attrGet('class')) {
									token.attrSet('class', 'animated-el');
								} else {
									const existingClass = token.attrGet('class');
									token.attrSet('class', existingClass + ' animated-el');
								}
							}
						}

						if (token.type.endsWith('_close')) {
							const tagName = token.tag;
							if (inlineElements.includes(tagName)) {
								insideInlineElement = false;
							}
						}
					});
				});
			});
		}

		md.core.ruler.after('inline', 'convert_elements', function (state) {
			state.tokens.forEach(function (blockToken) {
				if (blockToken.type !== 'inline') return;

				let fullContent = '';

				blockToken.children.forEach(token => {
					let { content, type } = token;

					switch (type) {
						case 'link_open':
							content = `<a ${token.attrs.map(([ key, value ]) => `${key}="${value}"`).join(' ')}>`;
							break;
						case 'link_close':
							content = '</a>';
							break;
					}

					fullContent += content;
				});

				if (
					fullContent.includes('<ol>') ||
					fullContent.includes('<ul>')
				) {
					const listToken = new state.Token('html_inline', '', 0);
					listToken.content = fullContent.trim();
					listToken.markup = 'html';
					listToken.type = 'html_inline';

					blockToken.children = [ listToken ];
				}
			});
		});

		md.core.ruler.after('inline', 'convert_links', function (state) {
			state.tokens.forEach(function (blockToken) {
				if (blockToken.type !== 'inline') return;
				blockToken.children.forEach(function (token, idx) {
					const { content } = token;
					if (content.includes('<a ')) {
						const linkRegex = /(.*)(<a\s+[^>]*\s+href="([^"]+)"[^>]*>([^<]*)<\/a>?)(.*)/;
						const linkMatch = content.match(linkRegex);

						if (linkMatch) {
							const [ , before, , href, text, after ] = linkMatch;

							const beforeToken = new state.Token('text', '', 0);
							beforeToken.content = before;

							const newToken = new state.Token('link_open', 'a', 1,);
							newToken.attrs = [
								[ 'href', href ],
								[ 'target', '_blank' ],
							];
							const textToken = new state.Token('text', '', 0);
							textToken.content = text;
							const closingToken = new state.Token('link_close', 'a', -1,);

							const afterToken = new state.Token('text', '', 0);
							afterToken.content = after;

							blockToken.children.splice(idx, 1, beforeToken, newToken, textToken, closingToken, afterToken,);
						}
					}
				});
			});
		});
	});

	// Add a renderer rule to handle emphasize and strong markup at the end of a string without closing markers
	renderer.use( function ( md ) {
		md.core.ruler.after( 'inline', 'fix_unclosed_markup', function ( state ) {
			state.tokens.forEach( function ( blockToken ) {
				if ( blockToken.type !== 'inline' ) return;

				blockToken.children.forEach( ( token, idx ) => {
					const { content } = token;

					// Check for unclosed markup at the end of the content
					if ( token.type === 'text' ) {
						// Replace multiple patterns in sequence
						let newContent = content;

						// Remove trailing *** (three or more asterisks)
						newContent = newContent.replace( /\*{3,}$/, '' );

						// Update content if modified
						if ( newContent !== content ) {
							token.content = newContent;
						}
					}
				} );
			} );
		} );
	} );

	let renderedString = renderer.render(renderer.utils.unescapeAll(string));

	// If the response contains a skill card, keep only the last one and strip trailing content.
	if (renderedString.includes('lqd-skill-card') || renderedString.includes('lqd-skill-card-shimmer')) {
		// Find all skill card blocks (wrapped in <pre style="display:contents;all:unset">)
		var skillCardPattern = /<pre style="display:contents;all:unset">\s*<div class="lqd-skill-card[^"]*"[\s\S]*?<\/pre>/g;
		var matches = renderedString.match(skillCardPattern);
		if (matches && matches.length > 0) {
			var lastCard = matches[matches.length - 1];
			// Find position of first skill card to keep content before it
			var firstCardPos = renderedString.indexOf(matches[0]);
			var beforeCards = renderedString.substring(0, firstCardPos);
			renderedString = beforeCards + lastCard;
		}
	}

	return renderedString;
}

const throttledRefreshFsLightbox = _.throttle(() => {
	if ('refreshFsLightbox' in window) {
		refreshFsLightbox();
	}
}, 250);

/**
 * Process smart image containers in the chat content.
 * Finds hidden containers with JSON image data and renders image grids.
 */
function validateImageUrl(url) {
	return new Promise(resolve => {
		const img = new Image();
		img.onload = () => resolve(true);
		img.onerror = () => resolve(false);
		img.src = url;
	});
}

async function buildSmartImageGridAsync(images) {
	// Validate all image URLs in parallel
	const results = await Promise.all(
		images.map(async img => {
			const url = img.imageUrl || img.thumbnailUrl;
			if (!url) return null;
			const valid = await validateImageUrl(url);
			return valid ? img : null;
		})
	);
	const validImages = results.filter(Boolean);

	return _buildSmartImageGridFromValidated(validImages);
}

function buildSmartImageGrid(images) {
	// Build grid immediately with shimmer, then replace broken images async
	const gridEl = _buildSmartImageGridFromValidated(images);

	// Async validation: remove broken items and rebuild if needed
	Promise.all(
		images.map(async img => {
			const url = img.imageUrl || img.thumbnailUrl;
			if (!url) return null;
			const valid = await validateImageUrl(url);
			return valid ? img : null;
		})
	).then(results => {
		const validImages = results.filter(Boolean);
		if (validImages.length < images.length && validImages.length > 0) {
			const newGrid = _buildSmartImageGridFromValidated(validImages);
			gridEl.replaceWith(newGrid);
		} else if (validImages.length === 0) {
			gridEl.remove();
		}
	});

	return gridEl;
}

function _buildSmartImageGridFromValidated(images) {
	const totalCount = images.length;
	const displayImages = images.slice(0, 3);
	// const displayCount = displayImages.length;
	// const cols = Math.min(displayCount, 3);
	const hasMore = totalCount > 3;

	const gridEl = document.createElement('div');
	gridEl.className = 'lqd-smart-image-grid relative mb-3 grid cursor-pointer grid-cols-3 gap-2';
	gridEl.dataset.smartImages = JSON.stringify(images);

	// Collect all image data for the lightbox gallery
	const allImageData = images.map(img => ({
		url: img.imageUrl || img.thumbnailUrl,
		title: img.title || '',
		source: img.source || '',
		domain: img.domain || '',
		link: img.link || '',
	}));

	// Show only first 3 images in the grid
	displayImages.forEach((img, index) => {
		const itemEl = document.createElement('div');
		itemEl.className = 'lqd-smart-image-item lqd-shimmer-effect aspect-[4/3] overflow-hidden rounded-[10px] bg-foreground/5';
		itemEl.dataset.index = index;

		const linkEl = document.createElement('a');
		linkEl.href = img.imageUrl || img.thumbnailUrl;
		linkEl.className = 'block size-full';

		const imgEl = document.createElement('img');
		imgEl.className = 'size-full object-cover';
		imgEl.alt = img.title || '';
		imgEl.loading = 'lazy';

		imgEl.addEventListener('load', () => {
			itemEl.classList.remove('lqd-shimmer-effect');
		});
		imgEl.addEventListener('error', () => {
			itemEl.classList.remove('lqd-shimmer-effect');
		});

		imgEl.src = img.thumbnailUrl || img.imageUrl;
		linkEl.appendChild(imgEl);
		itemEl.appendChild(linkEl);

		// Open lightbox on click
		linkEl.addEventListener('click', e => {
			e.preventDefault();
			openSmartImageLightbox(allImageData, index);
		});

		gridEl.appendChild(itemEl);

		if (hasMore) {
			const badge = document.createElement('span');
			badge.className = 'absolute bottom-2.5 end-2.5 inline-flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-2 text-2xs font-medium leading-none text-white backdrop-blur-sm';
			badge.innerHTML = `<svg class="size-3" width="12" height="12" viewBox="0 0 12 12" fill="currentColor" xmlns="http://www.w3.org/2000/svg" > <path d="M11 4H1C0.734784 4 0.48043 4.10536 0.292893 4.29289C0.105357 4.48043 0 4.73478 0 5V11C0 11.2652 0.105357 11.5196 0.292893 11.7071C0.48043 11.8946 0.734784 12 1 12H11C11.2652 12 11.5196 11.8946 11.7071 11.7071C11.8946 11.5196 12 11.2652 12 11V5C12 4.73478 11.8946 4.48043 11.7071 4.29289C11.5196 4.10536 11.2652 4 11 4ZM11 11H1V5H11V11ZM1 2.5C1 2.36739 1.05268 2.24021 1.14645 2.14645C1.24021 2.05268 1.36739 2 1.5 2H10.5C10.6326 2 10.7598 2.05268 10.8536 2.14645C10.9473 2.24021 11 2.36739 11 2.5C11 2.63261 10.9473 2.75979 10.8536 2.85355C10.7598 2.94732 10.6326 3 10.5 3H1.5C1.36739 3 1.24021 2.94732 1.14645 2.85355C1.05268 2.75979 1 2.63261 1 2.5ZM2 0.5C2 0.367392 2.05268 0.240215 2.14645 0.146447C2.24021 0.0526784 2.36739 0 2.5 0H9.5C9.63261 0 9.75979 0.0526784 9.85355 0.146447C9.94732 0.240215 10 0.367392 10 0.5C10 0.632608 9.94732 0.759785 9.85355 0.853553C9.75979 0.947321 9.63261 1 9.5 1H2.5C2.36739 1 2.24021 0.947321 2.14645 0.853553C2.05268 0.759785 2 0.632608 2 0.5Z" /> </svg>${totalCount}`;

			gridEl.appendChild(badge);
		}
	});

	return gridEl;
}

/**
 * Custom smart image lightbox using <img> tags to avoid CORS issues with external images.
 */
const smartImageLightbox = {
	el: null,
	imgEl: null,
	counterEl: null,
	prevBtn: null,
	nextBtn: null,
	images: [],
	index: 0,

	create() {
		if (this.el) return;

		this.el = document.createElement('div');
		this.el.id = 'lqd-smart-image-lightbox';
		// Use inline styles to avoid Tailwind purge issues with dynamic elements
		Object.assign(this.el.style, {
			position: 'fixed', inset: '0', zIndex: '99999', display: 'none',
			flexDirection: 'column'
		});
		this.el.innerHTML = `
			<div style="position:absolute;inset:0;background:rgba(0,0,0,0.8)" data-sil-close></div>
			<div style="position:relative;z-index:10;display:flex;flex-direction:column;height:100%">
				<div style="display:flex;align-items:center;justify-content:space-between;padding:16px;flex-shrink:0">
					<span style="font-size:14px;color:rgba(255,255,255,0.8)" data-sil-counter></span>
					<button style="width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,0.1);border:none;color:white;font-size:24px;cursor:pointer" data-sil-close>&times;</button>
				</div>
				<div style="flex:1;display:flex;align-items:center;justify-content:center;padding:0 16px 16px;overflow:hidden;position:relative">
					<button style="position:absolute;left:16px;top:50%;transform:translateY(-50%);z-index:10;width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,0.2);backdrop-filter:blur(8px);border:none;display:flex;align-items:center;justify-content:center;color:white;cursor:pointer" data-sil-prev>
						<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m15 18-6-6 6-6"/></svg>
					</button>
					<img style="max-height:100%;max-width:100%;object-fit:contain;border-radius:8px;transition:opacity .15s ease" data-sil-img />
					<button style="position:absolute;right:16px;top:50%;transform:translateY(-50%);z-index:10;width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,0.2);backdrop-filter:blur(8px);border:none;display:flex;align-items:center;justify-content:center;color:white;cursor:pointer" data-sil-next>
						<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>
					</button>
					<div style="position:absolute;bottom:16px;left:16px;z-index:20;max-width:400px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);border-radius:12px;padding:10px 14px;color:white;transition:opacity .15s ease" data-sil-source-info>
						<a style="display:flex;align-items:center;gap:6px;font-size:13px;font-weight:600;color:white;text-decoration:none;margin-bottom:2px" data-sil-source-link target="_blank" rel="noopener noreferrer">
							<img style="width:16px;height:16px;border-radius:50%" data-sil-source-favicon />
							<span data-sil-source-name></span>
						</a>
						<div style="font-size:12px;color:rgba(255,255,255,0.7);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" data-sil-source-title></div>
					</div>
				</div>
			</div>
		`;

		document.body.appendChild(this.el);

		this.imgEl = this.el.querySelector('[data-sil-img]');
		this.counterEl = this.el.querySelector('[data-sil-counter]');
		this.prevBtn = this.el.querySelector('[data-sil-prev]');
		this.nextBtn = this.el.querySelector('[data-sil-next]');
		this.sourceInfoEl = this.el.querySelector('[data-sil-source-info]');
		this.sourceLinkEl = this.el.querySelector('[data-sil-source-link]');
		this.sourceFaviconEl = this.el.querySelector('[data-sil-source-favicon]');
		this.sourceNameEl = this.el.querySelector('[data-sil-source-name]');
		this.sourceTitleEl = this.el.querySelector('[data-sil-source-title]');

		this.el.querySelectorAll('[data-sil-close]').forEach(btn => btn.addEventListener('click', () => this.close()));
		this.prevBtn.addEventListener('click', e => { e.stopPropagation(); this.nav(-1); });
		this.nextBtn.addEventListener('click', e => { e.stopPropagation(); this.nav(1); });

		document.addEventListener('keydown', e => {
			if (this.el.style.display === 'none') return;
			if (e.key === 'Escape') this.close();
			if (e.key === 'ArrowLeft') this.nav(-1);
			if (e.key === 'ArrowRight') this.nav(1);
		});
	},

	open(imageUrls, startIndex) {
		this.create();
		this.images = imageUrls;
		this.index = startIndex || 0;
		this.el.style.display = 'flex';
		document.body.style.overflow = 'hidden';
		this.render();
	},

	close() {
		this.el.style.display = 'none';
		document.body.style.overflow = '';
	},

	nav(dir) {
		const next = this.index + dir;
		if (next >= 0 && next < this.images.length) {
			this.index = next;
			this.render();
		}
	},

	render() {
		const current = this.images[this.index];
		const url = typeof current === 'string' ? current : (current.url || '');
		const title = typeof current === 'object' ? (current.title || '') : '';
		const source = typeof current === 'object' ? (current.source || current.domain || '') : '';
		const domain = typeof current === 'object' ? (current.domain || '') : '';
		const link = typeof current === 'object' ? (current.link || '') : '';

		// Hide old image and source info immediately while new image loads
		this.imgEl.style.opacity = '0';
		if (this.sourceInfoEl) this.sourceInfoEl.style.opacity = '0';

		this.counterEl.textContent = `${this.index + 1} / ${this.images.length}`;
		this.prevBtn.style.display = this.index > 0 ? 'flex' : 'none';
		this.nextBtn.style.display = this.index < this.images.length - 1 ? 'flex' : 'none';

		const showImage = () => {
			this.imgEl.style.opacity = '1';
			if (this.sourceInfoEl) this.sourceInfoEl.style.opacity = '1';
			this._preloadAdjacent();
		};

		this.imgEl.onload = showImage;
		this.imgEl.onerror = showImage;
		this.imgEl.src = url;

		if (this.sourceInfoEl) {
			if (source || title) {
				this.sourceInfoEl.style.display = 'block';
				this.sourceNameEl.textContent = source;
				this.sourceTitleEl.textContent = title;
				if (link) {
					this.sourceLinkEl.href = link;
					this.sourceLinkEl.style.pointerEvents = 'auto';
				} else {
					this.sourceLinkEl.removeAttribute('href');
					this.sourceLinkEl.style.pointerEvents = 'none';
				}
				if (domain) {
					this.sourceFaviconEl.src = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
					this.sourceFaviconEl.style.display = 'block';
				} else {
					this.sourceFaviconEl.style.display = 'none';
				}
			} else {
				this.sourceInfoEl.style.display = 'none';
			}
		}
	},

	_preloadAdjacent() {
		[ -1, 1 ].forEach(dir => {
			const idx = this.index + dir;
			if (idx < 0 || idx >= this.images.length) return;
			const img = this.images[idx];
			const src = typeof img === 'string' ? img : (img.url || '');
			if (src) new Image().src = src;
		});
	}
};

function openSmartImageLightbox(imageUrls, startIndex) {
	smartImageLightbox.open(imageUrls, startIndex);
}

function processSmartImageContainers(contentEl) {
	if (!contentEl) return;

	// Skip if a grid already exists inside this contentEl (prevents duplicates)
	if (contentEl.querySelector('.lqd-smart-image-grid')) {
		return;
	}

	const containers = contentEl.querySelectorAll('.lqd-smart-images-container');
	let gridInserted = false;

	containers.forEach(container => {
		// Skip if already processed
		if (container.dataset.processed) return;

		const text = container.textContent.trim();
		if (!text) return;

		try {
			const images = JSON.parse(text);
			if (!Array.isArray(images) || images.length === 0) return;

			container.dataset.processed = '1';

			const gridEl = buildSmartImageGrid(images);

			// Replace the hidden container with the grid
			container.replaceWith(gridEl);
			gridInserted = true;

			// Remove shimmer placeholder now that real images are shown
			const parentEl = gridEl.parentElement;
			if (parentEl) {
				const shimmer = parentEl.querySelector('.lqd-smart-images-shimmer');
				if (shimmer) shimmer.remove();
			}
		} catch (e) {
			// Not valid JSON yet (still streaming) — leave as-is
		}
	});

}

function hideTempNote() {
	const tempChatNote = document.getElementById('temp-chat-note');
	if (tempChatNote) {
		tempChatNote.style.display = 'none';
	}
}

function switchGenerateButtonsStatus(generating) {
	const generateBtn = document.querySelector('#send_message_button');
	const stopBtn = document.querySelector('#stop_button');
	const chatsWrapper = document.querySelector('.chats-wrap');

	chatsWrapper.classList.toggle('submitting', generating);

	generateBtn.disabled = generating;
	generateBtn.classList.toggle('hidden', generating);
	generateBtn.classList.toggle('submitting', generating);

	if (stopBtn) {
		stopBtn.classList.toggle('active', generating);
		stopBtn.disabled = !generating;
	}
}

/**
 *
 * @param {AiResponse} responseObj
 * @param {HTMLElement} el
 */
function setAnimatingWordY(responseObj, el) {
	if (!el || el?.classList?.contains('done-signal')) return;

	let { offsetTop } = el;

	if (offsetTop <= responseObj.lastAnimatedElOffsetTop) return;

	responseObj.lastAnimatedElOffsetTop = offsetTop;

	if (el.nodeName === 'TR') {
		offsetTop = offsetTop + (el.closest('table')?.offsetTop || 0);
	}

	responseObj.bubbleEl.style.setProperty('--animating-word-y', `${offsetTop}px`,);
}

/**
 *
 * @param {AiResponse} responseObj
 * @param {HTMLElement} el
 */
function onWordAnimationFinish(responseObj, el) {
	const isDoneSignal = el.classList.contains('done-signal');

	el.classList.replace('animating', 'animated');

	if (!responseObj.responseStreaming && isDoneSignal) {
		responseObj.bubbleEl.classList.replace('animating-words', 'animating-words-done');

		responseObj.animatingWordIndex = -1;

		switchGenerateButtonsStatus(aiResponses.every(res => res.responseStreaming));

		if ( responseObj.bubbleEl.querySelector('.social-media-agent-chat-post-card') ) {
			responseObj.bubbleEl.querySelector('.lqd-chat-bubble-canvas-trigger')?.remove();
			responseObj.bubbleEl.querySelectorAll('[data-copy-options],[data-copy-type]').forEach(el => el.remove());
		}

		_.defer(() => {
			unwrapWords(responseObj.chatContentEl);
			responseObj.chatContentEl.normalize();
			if (responseObj.entityHighlights) {
				applyEntityHighlights(responseObj.bubbleEl, responseObj.entityHighlights);
			}
		});
	}

	setAnimatingWordY(responseObj, el);
}

/**
 * @param {AiResponse} responseObj
 */
function animateNewElements(responseObj) {
	const allAnimatableElements = responseObj.chatContentEl.querySelectorAll('.animated-el, li, hr, tr');

	allAnimatableElements.forEach((el, index) => {
		// Skip if already animated
		if (responseObj.animatedElements.has(el)) {
			return;
		}

		// Mark as animated immediately
		responseObj.animatedElements.add(el);

		// Simple staggered animation with timeout
		setTimeout(() => {
			responseObj.bubbleEl.classList.replace('loading', 'streaming-started');

			el.animate([ { opacity: 1 } ], {
				duration: 500,
				easing: 'ease',
				fill: 'forwards',
			})
				.onfinish = onWordAnimationFinish(responseObj, el);
		}, index * 20);
	});
}

/**
 * Reset animation state for new responses
 * @param {AiResponse} responseObj
 */
function resetAnimationState(responseObj) {
	responseObj.animatedElements = new Set();
	responseObj.animatingWordIndex = -1;
}

/**
 * @param {AiResponse} responseObj
 */
function onAiResponse(responseObj) {
	const contentEl = responseObj.chatContentEl;
	let responseString = getAiResponseString({ responseObj, withoutDone: false });

	// Strip function call text, image annotations, entity annotations, and suggestions JSON
	responseString = cleanEntityAnnotations(cleanSmartImageLeaks(responseString))
		.replace(/\[search_images\([\s\S]*$/g, '')
		.replace(/search_images\{[\s\S]*$/g, '')
		.replace(/\s*(?:<br\s*\/?>|\n)*\s*\{[\s\n]*(?:<br\s*\/?>|\n)*\s*"suggestions"\s*:\s*\[[\s\S]*$/i, '');

	let formattedResponse = formatString(responseString, {
		readyForAnimation: true
	});

	// Prepend "Used X Skill" badge if skills were used
	if (responseObj._usedSkills?.length) {
		const existing = formattedResponse?.includes?.('lqd-skills-used');
		if (!existing) {
			let badge = '<div class="lqd-skills-used flex flex-wrap items-center gap-1.5 pt-3 text-xs text-foreground/50">';
			responseObj._usedSkills.forEach(function(skill) {
				badge += '<span>' + (magicai_localize.used ?? 'Used') + ' ' + escapeHtml(skill.name) + ' ' + (magicai_localize.skill ?? 'Skill') + '</span>';
			});
			badge += '</div>';
			formattedResponse = badge + (formattedResponse || '');
		}
	}

	// When the response contains a social media post card, bypass the VDOM diff
	// and use direct innerHTML so Alpine can properly initialize the component.
	// The VDOM's cloneNode approach prevents Alpine from detecting new x-data elements.
	const hasPostCard = formattedResponse && formattedResponse.includes('social-media-agent-chat-post-card');

	if (hasPostCard && !responseObj.responseStreaming) {
		contentEl.innerHTML = formattedResponse;

		// Post card has no .animated-el elements, so animateNewElements() will never
		// replace the bubble's 'loading' class with 'streaming-started'. Without this,
		// .chat-content stays display:none due to the .loading CSS rule.
		responseObj.bubbleEl.classList.remove('loading');
		responseObj.bubbleEl.classList.add('streaming-started');
	} else {
		const responseHTML = liquidVDOM.parseHTML(formattedResponse);

		liquidVDOM.clean(contentEl);
		liquidVDOM.diff(responseHTML, contentEl);
	}

	responseObj.bubbleEl.classList.toggle('streaming-on', responseObj.responseStreaming);

	animateNewElements(responseObj);

	// Process smart image containers (only when not streaming - during streaming
	// images are rendered directly in DOM outside the VDOM zone to prevent blinking)
	if (!responseObj.responseStreaming) {
		processSmartImageContainers(contentEl);
	}

	throttledRefreshFsLightbox();

	switchGenerateButtonsStatus(aiResponses.every(res => res.responseStreaming));
}

/**
 *
 * @param {AiResponse} responseObj
 */
async function handleCanvasResponseStore(responseObj) {
	try {
		const res = await fetch('/tiptap-content-store', {
			method: 'post',
			headers: {
				'Accept': 'application/json',
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				'message_id': responseObj.responseId,
				'content': formatString(getAiResponseString({ responseObj })),
				'type': 'output'
			})
		});
		const resData = await res.json();

		if (res.status === 401) {
			return;
		}

		if (!res.ok || resData.status == 'error') {
			toastr.error(resData.message || magicai_localize.could_not_save_the_canvas_data);
		}
	} catch (error) {
		alert(error);
		console.error(error);
	}
}

function onBeforePageUnload(e) {
	e.preventDefault();
	e.returnValue = '';
}

/**
 * @param {Event} event
 */
async function onAcceptResponseButtonClick(event) {
	event.preventDefault();

	const button = event.currentTarget;
	const messageId = button.getAttribute('data-message-id');
	const model = button.getAttribute('data-model');
	const bubbleEl = document.querySelector(`.lqd-chat-ai-bubble[data-message-id="${messageId}"]`);
	const multiAiResposeWrap = bubbleEl.closest('.multi-model-response-wrap');

	const formData = new FormData();
	formData.append('_token', document.querySelector('input[name=_token]')?.value);
	formData.append('messageId', messageId);

	await fetch(
		'/dashboard/user/multimodel/accept-response',
		{
			method: 'POST',
			body: formData
		}
	);

	multiAiResposeWrap.parentNode.insertBefore(bubbleEl, multiAiResposeWrap);
	multiAiResposeWrap.remove();

	const chatModelChangeEvent = new CustomEvent('chat-model-change', {
		detail: { model }
	});
	document.dispatchEvent(chatModelChangeEvent);
}

/**
 * @param {Event} event
 */
async function onRegenerateResponseButtonClick(event) {
	event.preventDefault();

	const button = event.currentTarget;
	const messageId = button.getAttribute('data-message-id');
	const model = button.getAttribute('data-model');

	console.log(event);
}

function getFrontModelEl() {
	let chatbotFrontModel = document.querySelector('#chatbot_front_model');

	if (!chatbotFrontModel) {
		chatbotFrontModel = document.createElement('select');
		chatbotFrontModel.id = 'chatbot_front_model';
		chatbotFrontModel.style.display = 'none';

		const defaultOption = document.createElement('option');
		defaultOption.value = '';
		defaultOption.textContent = magicai_localize.default_model || 'Default Model';
		chatbotFrontModel.appendChild(defaultOption);

		document.body.appendChild(chatbotFrontModel);
	}

	return chatbotFrontModel;
}

function getCouncilModeEl() {
	return document.querySelector('#chatbot_council_mode');
}

function isCouncilModeActive() {
	return getCouncilModeEl()?.value === '1';
}

function getModelLabelBySlug(slug = '') {
	if (!slug) return '';

	const modelSelect = getFrontModelEl();
	if (!modelSelect) return slug;

	const match = Array.from(modelSelect.options).find(option => option?.value === slug);
	if (!match) return slug;

	return `${match.innerText || ''}`.replace(/\n/g, '').trim() || slug;
}

function renderCouncilList(listEl, items = []) {
	if (!listEl) return;

	listEl.innerHTML = '';

	(items || []).forEach(item => {
		const li = document.createElement('li');
		li.classList.add('list-disc', 'ms-4');
		li.textContent = item;
		listEl.append(li);
	});
}

function normalizeCouncilListItems(items) {
	if (Array.isArray(items)) {
		return items
			.map(item => `${item ?? ''}`.trim())
			.filter(Boolean);
	}

	if (items && typeof items === 'object') {
		return Object.values(items)
			.map(item => `${item ?? ''}`.trim())
			.filter(Boolean);
	}

	if (typeof items === 'string') {
		return items
			.split(/\r?\n/)
			.map(item => item.replace(/^[-*•\d.)\s]+/, '').trim())
			.filter(Boolean);
	}

	return [];
}

function normalizeCouncilTableRows(rows) {
	const items = Array.isArray(rows) ? rows : (rows && typeof rows === 'object' ? Object.values(rows) : []);

	return items
		.filter(row => row && typeof row === 'object')
		.map(row => ({
			level: `${row?.level ?? ''}`.trim(),
			impact: `${row?.impact ?? ''}`.trim(),
		}))
		.filter(row => row.level || row.impact);
}

function getCouncilSectionTitleEl(responseObj, section) {
	const map = {
		final_answer: '.model-council-title-final-answer',
		agreement_analysis: '.model-council-title-agreement-analysis',
		disagreements: '.model-council-title-disagreements',
		discoveries: '.model-council-title-discoveries',
		model_replies: '.model-council-title-model-replies',
		agreement_level: '.model-council-title-agreement-level',
		confidence_impact: '.model-council-title-confidence-impact',
	};

	return responseObj?.bubbleEl?.querySelector(map[section] || '');
}

function setCouncilSectionTitle(responseObj, section, text = '') {
	const titleEl = getCouncilSectionTitleEl(responseObj, section);
	if (!titleEl) return;

	titleEl.textContent = text;
}

function setCouncilSummaryThinking(responseObj, visible = false) {
	const thinkingEl = responseObj?.bubbleEl?.querySelector('.model-council-summary-thinking');
	if (!thinkingEl) return;

	thinkingEl.classList.toggle('hidden', !visible);

	if (visible) {
		responseObj?.bubbleEl?.classList?.replace('loading', 'streaming-started');
	}
}

function setCouncilMeta(responseObj, meta = {}) {
	const respondedCount = Number(meta?.responded_models_count ?? 0);
	const confidencePercent = Number(meta?.confidence_percent ?? 0);
	const labels = meta?.labels || {};
	const respondedLabel = `${labels?.responded_models || ''}`.trim();
	const confidenceLabel = `${labels?.confidence || ''}`.trim();

	const respondedEl = responseObj?.bubbleEl?.querySelector('.model-council-responded-models');
	const confidenceEl = responseObj?.bubbleEl?.querySelector('.model-council-confidence');
	const sepEl = responseObj?.bubbleEl?.querySelector('.model-council-meta-sep');
	const metaLineEl = responseObj?.bubbleEl?.querySelector('.model-council-meta-line');

	if (respondedEl) {
		respondedEl.textContent = respondedCount > 0 && respondedLabel ? `${respondedCount} ${respondedLabel}` : '';
	}

	if (confidenceEl) {
		confidenceEl.textContent = confidencePercent >= 0 && confidenceLabel ? `${confidenceLabel} ${confidencePercent}%` : '';
	}

	if (sepEl) {
		sepEl.classList.toggle('hidden', !(respondedEl?.textContent && confidenceEl?.textContent));
	}

	if (metaLineEl) {
		metaLineEl.classList.toggle('hidden', !respondedEl?.textContent && !confidenceEl?.textContent);
	}
}

function getCouncilListEl(responseObj, section) {
	const map = {
		agreement_analysis: '.model-council-agreement-analysis',
		disagreements: '.model-council-disagreements',
		discoveries: '.model-council-discoveries',
	};

	return responseObj?.bubbleEl?.querySelector(map[section] || '');
}

function appendCouncilListItem(responseObj, section, item) {
	const listEl = getCouncilListEl(responseObj, section);
	if (!listEl || !item) return;

	const li = document.createElement('li');
	li.classList.add('list-disc', 'ms-4');
	li.textContent = `${item}`.trim();

	if (!li.textContent) {
		return;
	}

	listEl.append(li);
}

function clearCouncilLists(responseObj) {
	if (!responseObj?.bubbleEl) return;

	[ 'agreement_analysis', 'disagreements', 'discoveries' ].forEach(section => {
		const listEl = getCouncilListEl(responseObj, section);
		if (listEl) {
			listEl.innerHTML = '';
		}
	});
}

function clearCouncilTableRows(responseObj) {
	const rowsWrap = responseObj?.bubbleEl?.querySelector('.model-council-agreement-rows');
	if (!rowsWrap) return;

	rowsWrap.innerHTML = '';
}

function appendCouncilTableRow(responseObj, level = '', impact = '') {
	const rowsWrap = responseObj?.bubbleEl?.querySelector('.model-council-agreement-rows');
	if (!rowsWrap) return;

	const row = document.createElement('div');
	row.classList.add('grid', 'grid-cols-2', 'px-4', 'py-2', 'text-2xs');
	if (rowsWrap.children.length > 0) {
		row.classList.add('border-t', 'border-heading-foreground/10');
	}

	const levelEl = document.createElement('span');
	levelEl.textContent = level;
	const impactEl = document.createElement('span');
	impactEl.textContent = impact;

	row.append(levelEl, impactEl);
	rowsWrap.append(row);
}

function resetCouncilFinalAnswer(responseObj) {
	responseObj.response = [];

	if (responseObj.request) {
		responseObj.request.finalAnswer = '';
	}
}

function appendCouncilFinalAnswerChunk(responseObj, chunk = '') {
	if (!responseObj || typeof chunk !== 'string') return;

	const current = `${responseObj.request?.finalAnswer || ''}`;
	const spacer = current === '' ? '' : ' ';
	const nextText = `${current}${spacer}${chunk}`.trim();

	if (!responseObj.request) {
		responseObj.request = {};
	}

	responseObj.request.finalAnswer = nextText;
	responseObj.response = [ nextText ];
}

function prepareCouncilBubbleForStream(responseObj) {
	if (!responseObj?.bubbleEl) return;

	resetCouncilFinalAnswer(responseObj);
	setCouncilFinalSectionsVisible(responseObj, false);
	setCouncilSummaryThinking(responseObj, false);
	setCouncilMeta(responseObj, {});
	clearCouncilLists(responseObj);
	clearCouncilTableRows(responseObj);
	setCouncilSectionTitle(responseObj, 'final_answer', '');
	setCouncilSectionTitle(responseObj, 'agreement_analysis', '');
	setCouncilSectionTitle(responseObj, 'disagreements', '');
	setCouncilSectionTitle(responseObj, 'discoveries', '');
	setCouncilSectionTitle(responseObj, 'agreement_level', '');
	setCouncilSectionTitle(responseObj, 'confidence_impact', '');
	setCouncilSectionTitle(responseObj, 'model_replies', '');
	const repliesContainerEl = responseObj?.bubbleEl?.querySelector('.model-council-replies');
	if (repliesContainerEl) {
		repliesContainerEl.innerHTML = '';
	}
}

function setCouncilFinalSectionsVisible(responseObj, visible = false) {
	if (!responseObj?.bubbleEl) return;

	responseObj.bubbleEl.querySelectorAll('.model-council-final-sections').forEach(section => {
		section.classList.toggle('hidden', !visible);
	});
}

function finalizeCouncilBubbleAnimationState(responseObj) {
	if (!responseObj?.bubbleEl) return;

	responseObj.bubbleEl.classList.remove('animating-words', 'loading', 'streaming-on');
	responseObj.bubbleEl.classList.add('animating-words-done');
}

function ensureCouncilReplyCard(responseObj, reply = {}) {
	const repliesContainerEl = responseObj?.bubbleEl?.querySelector('.model-council-replies');
	if (!repliesContainerEl) return null;

	const modelSlug = `${reply?.model_slug || ''}`;
	const modelLabel = reply?.model_label || modelSlug || 'Model';
	let card = Array.from(repliesContainerEl.children).find(el => el.getAttribute('data-model-slug') === modelSlug);

	if (!card) {
		card = document.createElement('div');
		card.classList.add('prose', 'prose-sm', 'w-full', 'max-w-none', 'rounded-[20px]', 'border', 'border-foreground/5', 'p-5.5', 'transition-border', 'dark:prose-invert');
		card.setAttribute('data-model-slug', modelSlug);

		const title = document.createElement('p');
		const titleSpan = document.createElement('span');

		title.classList.add('mb-3', 'flex', 'items-center', 'gap-4');
		titleSpan.classList.add('inline-block', 'max-w-full', 'truncate', 'text-[12px]', 'font-medium', 'underline', 'underline-offset-4');

		title.innerHTML = '<svg class="shrink-0" width="15" height="14" viewBox="0 0 15 14" fill="currentColor" xmlns="http://www.w3.org/2000/svg" > <path d="M4.76586 11.495L5.08728 11.4297C5.1773 11.4117 5.25828 11.363 5.31647 11.292C5.37466 11.221 5.40645 11.132 5.40645 11.0402C5.40645 10.9484 5.37466 10.8594 5.31647 10.7884C5.25828 10.7174 5.1773 10.6688 5.08728 10.6507L4.76586 10.5854C4.36954 10.505 4.00569 10.3097 3.71974 10.0237C3.43379 9.7378 3.23842 9.37397 3.15801 8.97767L3.09275 8.65626C3.07471 8.56625 3.02605 8.48525 2.95503 8.42706C2.88402 8.36888 2.79504 8.3371 2.70323 8.3371C2.61142 8.3371 2.52245 8.36888 2.45143 8.42706C2.38042 8.48525 2.33175 8.56625 2.3137 8.65626L2.24844 8.97767C2.16804 9.37397 1.97266 9.7378 1.68671 10.0237C1.40076 10.3097 1.03692 10.505 0.640595 10.5854L0.319189 10.6507C0.229171 10.6688 0.148173 10.7174 0.0899825 10.7884C0.0317923 10.8594 0 10.9484 0 11.0402C0 11.132 0.0317923 11.221 0.0899825 11.292C0.148173 11.363 0.229171 11.4117 0.319189 11.4297L0.640595 11.495C1.03692 11.5754 1.40076 11.7708 1.68671 12.0567C1.97266 12.3426 2.16804 12.7065 2.24844 13.1028L2.3137 13.4242C2.33175 13.5142 2.38042 13.5952 2.45143 13.6534C2.52245 13.7116 2.61142 13.7433 2.70323 13.7433C2.79504 13.7433 2.88402 13.7116 2.95503 13.6534C3.02605 13.5952 3.07471 13.5142 3.09275 13.4242L3.15801 13.1028C3.23842 12.7065 3.43379 12.3426 3.71974 12.0567C4.00569 11.7708 4.36954 11.5754 4.76586 11.495Z" /> <path d="M12.5567 5.67479L13.7396 5.43497C13.8576 5.41083 13.9637 5.34666 14.0399 5.25332C14.1161 5.15998 14.1577 5.04318 14.1577 4.92269C14.1577 4.80221 14.1161 4.68542 14.0399 4.59208C13.9637 4.49873 13.8576 4.43457 13.7396 4.41042L12.5567 4.1706C11.9869 4.05496 11.4637 3.77405 11.0526 3.36291C10.6414 2.95178 10.3605 2.42865 10.2449 1.85884L10.005 0.67604C9.98131 0.557759 9.91735 0.451342 9.82403 0.374886C9.73071 0.29843 9.61379 0.256653 9.49315 0.256653C9.37251 0.256653 9.25559 0.29843 9.16228 0.374886C9.06896 0.451342 9.00499 0.557759 8.98126 0.67604L8.74143 1.85884C8.62589 2.4287 8.345 2.95188 7.93384 3.36303C7.52267 3.77418 6.99947 4.05506 6.42959 4.1706L5.24674 4.41042C5.12869 4.43457 5.02259 4.49873 4.9464 4.59208C4.87022 4.68542 4.8286 4.80221 4.8286 4.92269C4.8286 5.04318 4.87022 5.15998 4.9464 5.25332C5.02259 5.34666 5.12869 5.41083 5.24674 5.43497L6.42959 5.67479C6.99947 5.79032 7.52267 6.07121 7.93384 6.48236C8.345 6.89351 8.62589 7.4167 8.74143 7.98656L8.98126 9.16936C9.00499 9.28764 9.06896 9.39404 9.16228 9.4705C9.25559 9.54695 9.37251 9.58874 9.49315 9.58874C9.61379 9.58874 9.73071 9.54695 9.82403 9.4705C9.91735 9.39404 9.98131 9.28764 10.005 9.16936L10.2449 7.98656C10.3605 7.41674 10.6414 6.89361 11.0526 6.48248C11.4637 6.07135 11.9869 5.79042 12.5567 5.67479Z" /> </svg>';
		titleSpan.textContent = modelLabel;

		const content = document.createElement('p');
		content.classList.add('m-0', 'model-council-reply-content');

		title.append(titleSpan);
		card.append(title, content);
		repliesContainerEl.append(card);
	} else {
		const title = card.querySelector('.model-council-reply-title');

		if (title && !title.textContent?.trim()) {
			title.textContent = modelLabel;
		}
	}

	return card;
}

function cleanSmartImageLeaks(text) {
	return text
		.replace(/\[search_images\([\s\S]*?\)\]\s*/g, '')
		.replace(/(?:search_images)?\{"images"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}\s*/g, '')
		.replace(/\bsearch_images\b\s*/g, '')
		.replace(/\[Image results fetched[^\]]*\]\s*/gi, '')
		.replace(/\s*(?:<br\s*\/?>|\n)*\s*\{[\s\n]*(?:<br\s*\/?>|\n)*\s*"suggestions"\s*:\s*\[[\s\S]*?\]\s*(?:<br\s*\/?>|\n)*\s*\}/gi, '');
}

function cleanEntityAnnotations(text) {
	// Strip everything from :::meta or :::entity-highlights to end of text.
	// The metadata block is always the last thing in the response — one simple cut.
	text = text.replace(/\s*(?:<br\s*\/?>|\n)*\s*:::\s*(?:meta|entity[- ]?highlights?)[\s\S]*$/i, '');

	// Clean up trailing <br/> whitespace left after removal
	text = text.replace(/(?:<br\s*\/?>)+\s*$/, '');

	return text;
}

/**
 * Apply entity highlight annotations to a rendered message bubble.
 * Walks text nodes (excluding code, pre, a, already-highlighted spans) and wraps
 * the first occurrence of each entity in a clickable highlight span.
 */
/**
 * Wait for word animation spans to be unwrapped, then apply entity highlights.
 * Polls every 200ms for up to 10s. Falls back to applying anyway after timeout.
 */
function waitForUnwrapThenHighlight(responseObj) {
	const bubbleEl = responseObj.bubbleEl;
	const entities = responseObj.entityHighlights;
	if (!bubbleEl || !entities?.length) return;

	function tryApply() {
		const chatContentEl = bubbleEl.querySelector('.chat-content');
		if (!chatContentEl) {
			setTimeout(tryApply, 100);
			return;
		}

		// Force unwrap any animated spans and normalize text nodes
		if (chatContentEl.querySelector('span.animated-el')) {
			unwrapWords(chatContentEl);
		}
		chatContentEl.normalize();

		applyEntityHighlights(bubbleEl, entities);
	}

	// Short delay to let the stream fully flush, then apply immediately
	setTimeout(tryApply, 200);
}

function applyEntityHighlights(bubbleEl, entities) {
	if (!entities || entities.length === 0) return;

	const chatContentEl = bubbleEl.querySelector('.chat-content');
	if (!chatContentEl) return;

	// Skip if highlights already applied (e.g. server-rendered or duplicate call)
	if (chatContentEl.querySelector('.lqd-entity-highlight')) return;

	const highlighted = new Set();

	for (const entity of entities) {
		if (highlighted.has(entity.text.toLowerCase())) continue;

		// Walk text nodes to find and wrap entity text
		let found = _highlightEntityInTextNodes(chatContentEl, entity, false);

		// Fallback: case-insensitive search if exact match not found
		if (!found) {
			found = _highlightEntityInTextNodes(chatContentEl, entity, true);
		}

		if (found) {
			highlighted.add(entity.text.toLowerCase());
		}
	}
}

/**
 * Walk text nodes inside a container and wrap the first occurrence of entity text in a highlight span.
 * Returns true if a match was found and wrapped.
 */
function _highlightEntityInTextNodes(container, entity, caseInsensitive) {
	const walker = document.createTreeWalker(
		container,
		NodeFilter.SHOW_TEXT,
		{
			acceptNode: function(node) {
				const parent = node.parentElement;
				if (!parent) return NodeFilter.FILTER_REJECT;
				// Skip code blocks, links, and existing highlights — but allow the .chat-content <pre> itself
				if (parent.closest('code, pre:not(.chat-content), a, .lqd-entity-highlight')) {
					return NodeFilter.FILTER_REJECT;
				}
				return NodeFilter.FILTER_ACCEPT;
			}
		}
	);

	const textNodes = [];
	while (walker.nextNode()) {
		textNodes.push(walker.currentNode);
	}

	for (const textNode of textNodes) {
		const content = textNode.textContent;
		let idx;

		if (caseInsensitive) {
			idx = content.toLowerCase().indexOf(entity.text.toLowerCase());
		} else {
			idx = content.indexOf(entity.text);
		}

		if (idx === -1) continue;

		const originalText = content.substring(idx, idx + entity.text.length);
		const before = content.substring(0, idx);
		const after = content.substring(idx + entity.text.length);

		const span = document.createElement('span');
		span.className = 'lqd-entity-highlight';
		span.dataset.entityText = entity.text;
		span.dataset.entityType = entity.type;
		span.dataset.entityConfidence = entity.confidence;
		span.dataset.entityContext = entity.context_snippet || '';
		span.setAttribute('role', 'button');
		span.setAttribute('tabindex', '0');
		span.setAttribute('title', 'Learn more about ' + entity.text);
		span.textContent = originalText;

		const parent = textNode.parentNode;
		if (before) {
			parent.insertBefore(document.createTextNode(before), textNode);
		}
		parent.insertBefore(span, textNode);
		if (after) {
			parent.insertBefore(document.createTextNode(after), textNode);
		}
		parent.removeChild(textNode);

		return true;
	}

	return false;
}

// ============================================================
// Entity Highlight – Delegated Handlers
// ============================================================

// Delegated hover handler — prefetch entity details via the Alpine drawer component.
document.addEventListener('mouseenter', e => {
	if (!e.target || typeof e.target.closest !== 'function') return;
	const span = e.target.closest('.lqd-entity-highlight');
	if (!span) return;

	const entityData = {
		text: span.dataset.entityText,
		type: span.dataset.entityType,
		confidence: parseFloat(span.dataset.entityConfidence),
		context_snippet: span.dataset.entityContext || '',
	};
	const bubbleEl = span.closest('.lqd-chat-ai-bubble') || span.closest('[data-message-id]');
	if (bubbleEl) {
		document.dispatchEvent(new CustomEvent('entity-drawer:prefetch', {
			detail: { entity: entityData, messageId: bubbleEl.closest('[data-message-id]')?.dataset.messageId }
		}));
	}
}, true);

// Delegated click handler for entity highlights — works for both server-rendered and streaming spans.
document.addEventListener('click', e => {
	const span = e.target.closest('.lqd-entity-highlight');
	if (!span) return;

	e.preventDefault();
	e.stopPropagation();

	const entityData = {
		text: span.dataset.entityText,
		type: span.dataset.entityType,
		confidence: parseFloat(span.dataset.entityConfidence),
		context_snippet: span.dataset.entityContext || '',
	};
	const bubbleEl = span.closest('.lqd-chat-ai-bubble') || span.closest('[data-message-id]');
	if (bubbleEl) {
		document.dispatchEvent(new CustomEvent('entity-drawer:open', {
			detail: { entity: entityData, messageId: bubbleEl.closest('[data-message-id]')?.dataset.messageId }
		}));
	}
});

function setCouncilReplyTextContent(contentEl, text) {
	if (!contentEl) return;

	contentEl.innerHTML = formatString(text);
}

const throttledSetCouncilReplyTextContent = _.throttle(setCouncilReplyTextContent, 100);

function setCouncilReplyText(card, fullText) {
	if (!card) return;

	const content = card.querySelector('.model-council-reply-content');
	if (!content) return;

	const cleaned = cleanEntityAnnotations(cleanSmartImageLeaks(fullText));
	throttledSetCouncilReplyTextContent(content, cleaned);
	card.setAttribute('data-full-text', cleaned);
}

function finalizeCouncilReplyCard(card) {
	if (!card) return;

	const existingButton = card.querySelector('.model-council-reply-toggle');
	if (existingButton) {
		existingButton.remove();
	}

	const content = card.querySelector('.model-council-reply-content');
	const fullText = `${card.getAttribute('data-full-text') || ''}`;

	if (!content || fullText.length <= 320) {
		return;
	}

	const shortText = `${fullText.slice(0, 320)}...`;

	throttledSetCouncilReplyTextContent(content, shortText);

	const button = document.createElement('button');
	button.classList.add('model-council-reply-toggle', 'mt-4', 'text-3xs', 'font-medium', 'text-primary', 'border', 'rounded-full', 'px-3', 'py-1.5', 'hover:bg-primary', 'hover:text-primary-foreground', 'hover:border-primary');
	button.textContent = 'View Full Response';
	button.type = 'button';

	let expanded = false;
	button.addEventListener('click', () => {
		expanded = !expanded;

		throttledSetCouncilReplyTextContent(content, expanded ? fullText : shortText);

		button.textContent = expanded ? 'Show Less' : 'View Full Response';
	});

	card.append(button);
}

function appendCouncilReplyChunk(responseObj, modelSlug, modelLabel, chunk) {
	const card = ensureCouncilReplyCard(responseObj, {
		model_slug: modelSlug,
		model_label: modelLabel,
	});

	if (!card) return;

	const current = `${card.getAttribute('data-full-text') || ''}`;
	setCouncilReplyText(card, `${current}${chunk}`);
}

function registerCouncilAbortController(controller) {
	if (!controller) return;

	councilChildAbortControllers.push(controller);
}

function unregisterCouncilAbortController(controller) {
	if (!controller) return;

	councilChildAbortControllers = councilChildAbortControllers.filter(item => item !== controller);
}

function abortCouncilChildStreams() {
	councilChildAbortControllers.forEach(controller => controller?.abort?.());
	councilChildAbortControllers = [];
}

function sendCouncilModelRequest(type, modelSlug, sharedMessageUUID, councilResponseObj) {
	if (!modelSlug || !councilResponseObj?.bubbleEl) {

		return Promise.resolve(false);
	}

	const modelLabel = getModelLabelBySlug(modelSlug);

	return new Promise(resolve => {
		const formData = new FormData();
		const promptInput = document.getElementById('prompt');
		const chat_id = document.querySelector('#chat_id')?.value;
		const abortController = new AbortController();
		let receivedMessageId = false;
		let done = false;

		const finish = ok => {
			if (done) return;
			done = true;
			try { unregisterCouncilAbortController(abortController); } catch (_) {}
			try { if (!ok) abortController.abort(); } catch (_) {}
			resolve(ok);
		};

		// Hard timeout per model — no matter what, this will resolve
		const timeoutId = setTimeout(() => {
			if (!done) {

				appendCouncilReplyChunk(councilResponseObj, modelSlug, modelLabel, ' (timed out)');
				const card = ensureCouncilReplyCard(councilResponseObj, { model_slug: modelSlug, model_label: modelLabel });
				finalizeCouncilReplyCard(card);
				finish(false);
			}
		}, 90_000);

		const finishAndClear = ok => {
			clearTimeout(timeoutId);
			finish(ok);
		};

		formData.append('template_type', type);
		formData.append('prompt', buildPrompt(promptInput?.value || ''));
		formData.append('chat_id', chat_id || '');
		formData.append('category_id', category?.id);
		formData.append('images', '');
		formData.append('pdfname', '');
		formData.append('pdfpath', '');
		formData.append('realtime', document.getElementById('realtime')?.checked ? 1 : 0);
		formData.append('chat_brand_voice', document.getElementById('chat_brand_voice')?.value || '');
		formData.append('brand_voice_prod', document.getElementById('brand_voice_prod')?.value || '');
		formData.append('chatbot_front_model', modelSlug);
		formData.append('assistant', document.getElementById('assistant')?.value || '');

		if (document.querySelector('#chat_open_ai_agent_id')?.value) {
			formData.append('chat_open_ai_agent_id', document.querySelector('#chat_open_ai_agent_id').value);
		}
		if (sharedMessageUUID) {
			formData.append('shared_message_uuid', sharedMessageUUID);
		}
		formData.append('council_sub_request', '1');
		if (document.querySelector('#temp_chat_button')?.classList.contains('active')) {
			formData.append('temp_chat_button', '1');
		}

		const card = ensureCouncilReplyCard(councilResponseObj, { model_slug: modelSlug, model_label: modelLabel });
		if (card) setCouncilReplyText(card, '');

		registerCouncilAbortController(abortController);

		try {
			fetchEventSource('/dashboard/user/generator/generate-stream', {
				openWhenHidden: true,
				method: 'POST',
				headers: { 'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content') },
				body: formData,
				signal: abortController.signal,
				onmessage: async e => {
					if (done) return;
					if (e.event === 'chat_not_found') { window.location.reload(); return; }
					if (!receivedMessageId && e.event === 'message') { receivedMessageId = true; return; }
					if (!receivedMessageId) return;
					const txt = e.data;
					if (txt == null) return;

					if (txt.includes('[DONE]')) {

						const latestCard = ensureCouncilReplyCard(councilResponseObj, { model_slug: modelSlug, model_label: modelLabel });
						finalizeCouncilReplyCard(latestCard);
						finishAndClear(true);
						return;
					}

					// Skip non-content SSE events in council sub-requests
					if (e.event === 'smart_image_search' || e.event === 'entity_highlights' || e.event === 'suggestions' || e.event === 'title' || e.event === 'skills_used') {
						return;
					}

					appendCouncilReplyChunk(councilResponseObj, modelSlug, modelLabel, `${txt}`.replace(/<br\s*\/?>/g, '\n'));
				},
				onclose: () => {

					if (!done) {
						appendCouncilReplyChunk(councilResponseObj, modelSlug, modelLabel, 'Failed to Respond.');
						const latestCard = ensureCouncilReplyCard(councilResponseObj, { model_slug: modelSlug, model_label: modelLabel });
						finalizeCouncilReplyCard(latestCard);
						finishAndClear(false);
					}
				},
				onerror: err => {

					appendCouncilReplyChunk(councilResponseObj, modelSlug, modelLabel, 'Failed to Respond.');
					const latestCard = ensureCouncilReplyCard(councilResponseObj, { model_slug: modelSlug, model_label: modelLabel });
					finalizeCouncilReplyCard(latestCard);
					finishAndClear(false);
					throw err;
				},
			});
		} catch (err) {

			appendCouncilReplyChunk(councilResponseObj, modelSlug, modelLabel, 'Failed to Respond.');
			const latestCard = ensureCouncilReplyCard(councilResponseObj, { model_slug: modelSlug, model_label: modelLabel });
			finalizeCouncilReplyCard(latestCard);
			finishAndClear(false);
		}
	});
}

function streamCouncilModelRequests(selectedModelSlugs, sharedMessageUUID, councilResponseObj) {
	const requests = (selectedModelSlugs || []).map(modelSlug =>
		sendCouncilModelRequest('chatPro', modelSlug, sharedMessageUUID, councilResponseObj)
			.catch(() => false)
	);

	return Promise.allSettled(requests);
}

function renderCouncilReplies(containerEl, replies = []) {
	if (!containerEl) return;

	containerEl.innerHTML = '';

	(replies || []).forEach(reply => {
		const card = document.createElement('div');
		card.classList.add('prose', 'prose-sm', 'w-full', 'max-w-none', 'rounded-[20px]', 'border', 'border-foreground/5', 'p-5.5', 'transition-border', 'dark:prose-invert');
		card.setAttribute('data-model-slug', `${reply?.model_slug || ''}`);

		const title = document.createElement('p');
		const titleSpan = document.createElement('span');

		title.classList.add('mb-3', 'flex', 'items-center', 'gap-4');
		titleSpan.classList.add('inline-block', 'max-w-full', 'truncate', 'text-[12px]', 'font-medium', 'underline', 'underline-offset-4');

		title.innerHTML = '<svg class="shrink-0" width="15" height="14" viewBox="0 0 15 14" fill="currentColor" xmlns="http://www.w3.org/2000/svg" > <path d="M4.76586 11.495L5.08728 11.4297C5.1773 11.4117 5.25828 11.363 5.31647 11.292C5.37466 11.221 5.40645 11.132 5.40645 11.0402C5.40645 10.9484 5.37466 10.8594 5.31647 10.7884C5.25828 10.7174 5.1773 10.6688 5.08728 10.6507L4.76586 10.5854C4.36954 10.505 4.00569 10.3097 3.71974 10.0237C3.43379 9.7378 3.23842 9.37397 3.15801 8.97767L3.09275 8.65626C3.07471 8.56625 3.02605 8.48525 2.95503 8.42706C2.88402 8.36888 2.79504 8.3371 2.70323 8.3371C2.61142 8.3371 2.52245 8.36888 2.45143 8.42706C2.38042 8.48525 2.33175 8.56625 2.3137 8.65626L2.24844 8.97767C2.16804 9.37397 1.97266 9.7378 1.68671 10.0237C1.40076 10.3097 1.03692 10.505 0.640595 10.5854L0.319189 10.6507C0.229171 10.6688 0.148173 10.7174 0.0899825 10.7884C0.0317923 10.8594 0 10.9484 0 11.0402C0 11.132 0.0317923 11.221 0.0899825 11.292C0.148173 11.363 0.229171 11.4117 0.319189 11.4297L0.640595 11.495C1.03692 11.5754 1.40076 11.7708 1.68671 12.0567C1.97266 12.3426 2.16804 12.7065 2.24844 13.1028L2.3137 13.4242C2.33175 13.5142 2.38042 13.5952 2.45143 13.6534C2.52245 13.7116 2.61142 13.7433 2.70323 13.7433C2.79504 13.7433 2.88402 13.7116 2.95503 13.6534C3.02605 13.5952 3.07471 13.5142 3.09275 13.4242L3.15801 13.1028C3.23842 12.7065 3.43379 12.3426 3.71974 12.0567C4.00569 11.7708 4.36954 11.5754 4.76586 11.495Z" /> <path d="M12.5567 5.67479L13.7396 5.43497C13.8576 5.41083 13.9637 5.34666 14.0399 5.25332C14.1161 5.15998 14.1577 5.04318 14.1577 4.92269C14.1577 4.80221 14.1161 4.68542 14.0399 4.59208C13.9637 4.49873 13.8576 4.43457 13.7396 4.41042L12.5567 4.1706C11.9869 4.05496 11.4637 3.77405 11.0526 3.36291C10.6414 2.95178 10.3605 2.42865 10.2449 1.85884L10.005 0.67604C9.98131 0.557759 9.91735 0.451342 9.82403 0.374886C9.73071 0.29843 9.61379 0.256653 9.49315 0.256653C9.37251 0.256653 9.25559 0.29843 9.16228 0.374886C9.06896 0.451342 9.00499 0.557759 8.98126 0.67604L8.74143 1.85884C8.62589 2.4287 8.345 2.95188 7.93384 3.36303C7.52267 3.77418 6.99947 4.05506 6.42959 4.1706L5.24674 4.41042C5.12869 4.43457 5.02259 4.49873 4.9464 4.59208C4.87022 4.68542 4.8286 4.80221 4.8286 4.92269C4.8286 5.04318 4.87022 5.15998 4.9464 5.25332C5.02259 5.34666 5.12869 5.41083 5.24674 5.43497L6.42959 5.67479C6.99947 5.79032 7.52267 6.07121 7.93384 6.48236C8.345 6.89351 8.62589 7.4167 8.74143 7.98656L8.98126 9.16936C9.00499 9.28764 9.06896 9.39404 9.16228 9.4705C9.25559 9.54695 9.37251 9.58874 9.49315 9.58874C9.61379 9.58874 9.73071 9.54695 9.82403 9.4705C9.91735 9.39404 9.98131 9.28764 10.005 9.16936L10.2449 7.98656C10.3605 7.41674 10.6414 6.89361 11.0526 6.48248C11.4637 6.07135 11.9869 5.79042 12.5567 5.67479Z" /> </svg>';

		titleSpan.textContent = reply?.model_label || reply?.model_slug || 'Model';

		const fullText = reply?.failed ? 'Failed to Respond.' : (reply?.response_text || '');
		const shortText = fullText.length > 320 ? `${fullText.slice(0, 320)}...` : fullText;
		const content = document.createElement('p');

		content.classList.add('m-0', 'model-council-reply-content');
		content.textContent = shortText;
		card.setAttribute('data-full-text', fullText);

		title.append(titleSpan);
		card.append(title, content);

		if (fullText.length > 320) {
			const button = document.createElement('button');
			button.classList.add('mt-2', 'text-2xs', 'font-medium', 'text-primary', 'underline', 'underline-offset-4');
			button.textContent = 'View Full Response';
			button.type = 'button';

			let expanded = false;
			button.addEventListener('click', () => {
				expanded = !expanded;
				content.textContent = expanded ? fullText : shortText;
				button.textContent = expanded ? 'Show Less' : 'View Full Response';
			});

			card.append(button);
		}

		containerEl.append(card);
	});
}

function renderCouncilResponsePayload(responseObj, payload = {}) {
	if (!responseObj?.bubbleEl || !payload) return;

	const finalAnswer = payload?.final_answer || '';
	const councilResponse = payload?.council_response || {};
	const modelReplies = payload?.model_replies || [];
	const analysisListEl = responseObj.bubbleEl.querySelector('.model-council-agreement-analysis');
	const disagreementsListEl = responseObj.bubbleEl.querySelector('.model-council-disagreements');
	const discoveriesListEl = responseObj.bubbleEl.querySelector('.model-council-discoveries');
	const repliesContainerEl = responseObj.bubbleEl.querySelector('.model-council-replies');

	setCouncilMeta(responseObj, councilResponse?.meta || payload?.meta || {});

	const analysisItems = normalizeCouncilListItems(councilResponse?.agreement_analysis || payload?.agreement_analysis);
	const disagreementItems = normalizeCouncilListItems(councilResponse?.disagreements || payload?.disagreements);
	const discoveryItems = normalizeCouncilListItems(councilResponse?.unique_discoveries || payload?.unique_discoveries || payload?.discoveries);
	if (analysisItems.length > 0) {
		renderCouncilList(analysisListEl, analysisItems);
	}
	if (disagreementItems.length > 0) {
		renderCouncilList(disagreementsListEl, disagreementItems);
	}
	if (discoveryItems.length > 0) {
		renderCouncilList(discoveriesListEl, discoveryItems);
	}

	const tableRows = normalizeCouncilTableRows(councilResponse?.agreement_table_rows || payload?.agreement_table_rows);
	if (tableRows.length > 0) {
		clearCouncilTableRows(responseObj);
		tableRows.forEach(row => {
			appendCouncilTableRow(responseObj, row?.level ?? '', row?.impact ?? '');
		});
	}
	const titles = councilResponse?.titles || payload?.titles || {};
	if (titles.final_answer) setCouncilSectionTitle(responseObj, 'final_answer', titles.final_answer);
	if (titles.agreement_level) setCouncilSectionTitle(responseObj, 'agreement_level', titles.agreement_level);
	if (titles.confidence_impact) setCouncilSectionTitle(responseObj, 'confidence_impact', titles.confidence_impact);
	if (titles.agreement_analysis) setCouncilSectionTitle(responseObj, 'agreement_analysis', titles.agreement_analysis);
	if (titles.disagreements) setCouncilSectionTitle(responseObj, 'disagreements', titles.disagreements);
	if (titles.discoveries) setCouncilSectionTitle(responseObj, 'discoveries', titles.discoveries);
	if (titles.model_replies) setCouncilSectionTitle(responseObj, 'model_replies', titles.model_replies);
	if (Array.isArray(modelReplies) && modelReplies.length > 0) {
		renderCouncilReplies(repliesContainerEl, modelReplies);
	}

	setCouncilSummaryThinking(responseObj, false);
	setCouncilFinalSectionsVisible(responseObj, true);
	finalizeCouncilBubbleAnimationState(responseObj);
	if (responseObj.request) {
		responseObj.request.finalAnswer = finalAnswer;
	}

	responseObj.response = [ finalAnswer ];
}

function handleCouncilStreamPayload(responseObj, payload = {}) {
	if (!responseObj?.bubbleEl || !payload || typeof payload !== 'object') return false;

	const type = payload?.type;
	const suppressModelReplay = Boolean(responseObj?.request?.skipModelReplay);

	if (type === 'council_phase') {
		const phase = `${payload?.phase || ''}`;

		if (phase === 'collecting_models' || phase === 'synthesizing') {
			setCouncilSummaryThinking(responseObj, true);
			return false;
		}

		if (phase === 'final_streaming') {
			setCouncilSummaryThinking(responseObj, false);
			setCouncilFinalSectionsVisible(responseObj, true);
			clearCouncilLists(responseObj);
			clearCouncilTableRows(responseObj);
			resetCouncilFinalAnswer(responseObj);
			return true;
		}

		if (phase === 'done') {
			setCouncilSummaryThinking(responseObj, false);
			return false;
		}

		return false;
	}

	if (type === 'model_stream_start') {
		if (suppressModelReplay) {
			return false;
		}

		const reply = payload?.reply || {};
		const card = ensureCouncilReplyCard(responseObj, reply);
		if (card) {
			setCouncilReplyText(card, '');
		}
		return false;
	}

	if (type === 'model_stream_delta') {
		if (suppressModelReplay) {
			return false;
		}

		const modelSlug = `${payload?.model_slug || ''}`;
		const card = ensureCouncilReplyCard(responseObj, {
			model_slug: modelSlug,
			model_label: payload?.model_label || ''
		});
		const current = `${card?.getAttribute('data-full-text') || ''}`;
		const spacer = current === '' ? '' : ' ';
		const nextText = `${current}${spacer}${payload?.chunk || ''}`.trim();
		setCouncilReplyText(card, nextText);
		return false;
	}

	if (type === 'model_stream_end') {
		if (suppressModelReplay) {
			return false;
		}

		const modelSlug = `${payload?.model_slug || ''}`;
		const card = ensureCouncilReplyCard(responseObj, {
			model_slug: modelSlug,
			model_label: payload?.model_label || ''
		});
		finalizeCouncilReplyCard(card);
		return false;
	}

	if (type === 'summary_meta') {
		setCouncilMeta(responseObj, payload?.meta || {});
		return false;
	}

	if (type === 'summary_section_title') {
		const section = `${payload?.section || ''}`;

		setCouncilSectionTitle(responseObj, section, `${payload?.title || ''}`);

		return false;
	}

	if (type === 'summary_final_answer_delta') {
		appendCouncilFinalAnswerChunk(responseObj, `${payload?.chunk || ''}`);
		return true;
	}

	if (type === 'summary_table_row') {
		appendCouncilTableRow(responseObj, `${payload?.level || ''}`, `${payload?.impact || ''}`);
		return false;
	}

	if (type === 'summary_list_item') {
		appendCouncilListItem(responseObj, `${payload?.section || ''}`, `${payload?.item || ''}`);
		return true;
	}

	if (type === 'summary_done') {
		if (responseObj.request) {
			responseObj.request.finalAnswer = `${payload?.final_answer || responseObj.request.finalAnswer || ''}`;
		}

		if (payload?.council_response) {
			renderCouncilResponsePayload(responseObj, {
				final_answer: payload?.final_answer || responseObj.request?.finalAnswer || '',
				council_response: payload?.council_response || {},
			});
		}
		finalizeCouncilBubbleAnimationState(responseObj);

		return true;
	}

	if (type === 'error') {
		setCouncilSummaryThinking(responseObj, false);
		setCouncilFinalSectionsVisible(responseObj, true);
		responseObj.response = [ `${payload?.message || magicai_localize.error}` ];
		finalizeCouncilBubbleAnimationState(responseObj);
		return true;
	}

	if (type === 'final_summary' || payload?.final_answer) {
		renderCouncilResponsePayload(responseObj, payload);
		return true;
	}

	return false;
}

function createAiResponses(options = {}) {
	const { councilMode = false } = options;
	const aiBubbleTemplateEl = document.querySelector('#chat_ai_bubble');
	const councilBubbleTemplateEl = document.querySelector('#chat_ai_council_bubble');
	const canvasEditBtnTemplate = document.querySelector('#canvas_edit_btn_block');
	const canvasModeActivated = document.querySelector('#create_canvas_button.active');
	const chatsContainer = document.querySelector('.chats-container');
	const chatbotFrontModel = getFrontModelEl();
	const selectedOptions = Array.from(chatbotFrontModel.selectedOptions);
	const multiModelsSelected = !councilMode && selectedOptions.length > 1;
	let appendAiBubblesTo = chatsContainer;

	if (multiModelsSelected) {
		const multiAiResposeWrap = document.createElement('div');

		multiAiResposeWrap.classList.add('multi-model-response-wrap', 'grid', 'grid-cols-1', 'lg:grid-cols-2', 'gap-x-6');

		chatsContainer.insertAdjacentElement('beforeend', multiAiResposeWrap);

		appendAiBubblesTo = multiAiResposeWrap;
	}

	const optionsToRender = councilMode
		? [ selectedOptions[0] || chatbotFrontModel.options[0] ].filter(Boolean)
		: selectedOptions;

	optionsToRender.forEach(option => {
		const slug = option.value;
		const label = option.innerText.replace(/\n/g, '').trim();
		const bubbleTemplate = councilMode && councilBubbleTemplateEl ? councilBubbleTemplateEl : aiBubbleTemplateEl;
		const bubbleEl = bubbleTemplate.content.cloneNode(true).firstElementChild;
		const chatContentContainerEl = bubbleEl.querySelector('.chat-content-container');
		const chatContentEl = bubbleEl.querySelector('.chat-content');
		let acceptButtonEl = null;
		let regenerateButtonEl = null;

		bubbleEl.setAttribute('data-model', slug);
		bubbleEl.classList.add('loading', 'animating-words', multiModelsSelected ? 'w-full' : 'w-auto');

		if (category.slug === 'ai_chat_image') {
			bubbleEl.querySelector('.chat-content-container')?.classList?.add('flex', 'items-center');
			bubbleEl.querySelector('.lqd-typing')?.remove();
			bubbleEl.querySelector('button')?.remove();
		}

		if (multiModelsSelected) {
			const multiModelMessageHeadTemplate = document.querySelector('#multi-model-response-head');
			const multiModelMessageFootTemplate = document.querySelector('#multi-model-response-foot');

			if (multiModelMessageHeadTemplate) {
				const headEl = multiModelMessageHeadTemplate.content.cloneNode(true).firstElementChild;
				const nameEl = headEl.querySelector('.multi-model-response-name');

				nameEl.innerText = label;
				nameEl.setAttribute('title', label);

				chatContentContainerEl.insertAdjacentElement('afterbegin', headEl);
			}
			if (multiModelMessageFootTemplate) {
				const footEl = multiModelMessageFootTemplate.content.cloneNode(true).firstElementChild;
				chatContentContainerEl.insertAdjacentElement('beforeend', footEl);
			}

			acceptButtonEl = bubbleEl.querySelector('.multi-model-response-accept');
			regenerateButtonEl = bubbleEl.querySelector('.multi-model-response-regenerate');

			if (acceptButtonEl) {
				acceptButtonEl.addEventListener('click', onAcceptResponseButtonClick);
				acceptButtonEl.setAttribute('data-model', slug);
			}
			if (regenerateButtonEl) {
				regenerateButtonEl.addEventListener('click', onRegenerateResponseButtonClick);
				regenerateButtonEl.setAttribute('data-model', slug);
			}
		}

		if (canvasModeActivated && canvasEditBtnTemplate) {
			const canvasEditBtn = canvasEditBtnTemplate.content.cloneNode(true).firstElementChild;
			chatContentContainerEl.insertAdjacentElement('afterbegin', canvasEditBtn);
		}

		aiResponses.push({
			model: {
				slug,
				label
			},
			bubbleEl,
			chatContentEl,
			chatContentContainerEl,
			acceptButtonEl,
			regenerateButtonEl,
			responseStreaming: true,
			response: [],
			request: null,
			animatingWordIndex: -1,
			animatedElements: new Set(),
			lastAnimatedElOffsetTop: 0
		});

		appendAiBubblesTo.append(bubbleEl);
	});
}

/**
 *
 * @param {string} type
 * @param {any} images
 * @param {AiResponse} responseObj
 * @param {string | null} sharedMessageUUID
 * @param {{ mode?: string, councilModelSlugs?: string[], skipModelReplay?: boolean, preserveState?: boolean }} requestOptions
 */
function sendRequest(type, images, responseObj, sharedMessageUUID = null, requestOptions = {}) {
	const formData = new FormData();
	const tempChatButton = document.querySelector('#temp_chat_button');
	const realtime = document.getElementById('realtime');
	const chatBrandVoice = document.getElementById('chat_brand_voice');
	const brandVoiceProd = document.getElementById('brand_voice_prod');
	const assistant = document.getElementById('assistant');
	const canvasModeActivated = document.querySelector('#create_canvas_button.active');
	const promptInput = document.getElementById('prompt');
	const chat_id = document.querySelector('#chat_id')?.value;
	const throttledOnAiResponse = _.throttle(onAiResponse, 100);
	const abortController = new AbortController();
	const chatsV2 = Alpine.store('chatsV2');
	let receivedMessageId = false;

	formData.append('template_type', type);
	formData.append('prompt', buildPrompt(promptInput?.value));

	formData.append('chat_id', chat_id);
	formData.append('category_id', category?.id);
	formData.append('images', images == undefined ? '' : images);
	formData.append('pdfname', pdfName == undefined ? '' : pdfName);
	formData.append('pdfpath', pdfPath == undefined ? '' : pdfPath);
	formData.append('realtime', realtime?.checked ? 1 : 0);
	formData.append('chat_brand_voice', chatBrandVoice?.value || '');
	formData.append('brand_voice_prod', brandVoiceProd?.value || '');
	formData.append('chatbot_front_model', responseObj?.model.slug);
	formData.append('assistant', assistant?.value || '');
	const skillIdsInput = document.getElementById('selected_skill_ids');
	formData.append('skill_ids', skillIdsInput?.value || '');
	if (skillIdsInput) skillIdsInput.value = '';

	if (Array.isArray(requestOptions?.councilModelSlugs)) {
		requestOptions.councilModelSlugs.forEach(slug => {
			if (slug) {
				formData.append('council_model_slugs[]', slug);
			}
		});
	}

	if (requestOptions?.skipModelReplay) {
		formData.append('skip_model_replay', '1');
	}

	if (document.querySelector('#chat_open_ai_agent_id')?.value) {
		formData.append('chat_open_ai_agent_id', document.querySelector('#chat_open_ai_agent_id').value);
	}

	if (sharedMessageUUID) {
		formData.append('shared_message_uuid', sharedMessageUUID);
	}

	if (tempChatButton && tempChatButton.classList.contains('active')) {
		formData.append('temp_chat_button', '1');
	}

	if ( chatsV2 && chatsV2.selectedTools ) {
		const followUps = chatsV2.selectedTools.filter(tool => tool.startsWith('follow-up-'));

		if (followUps.length) {
			formData.append('highlight_context', followUps.map(f => f.replace('follow-up-', '')).join(', '));
		}
	}

	responseObj.abortController = abortController;
	responseObj.request = requestOptions || null;

	if (responseObj.request?.mode === 'council') {
		responseObj.request.finalAnswer = `${responseObj.request?.finalAnswer || ''}`;
		if (!responseObj.request?.preserveState) {
			prepareCouncilBubbleForStream(responseObj);
		}
	}

	if ( chatsV2 && chatsV2.selectedTools ) {
		chatsV2.selectedTools = chatsV2.selectedTools.filter(t => !t.startsWith('skill-') && !t.startsWith('follow-up-') && !t.startsWith('entity-follow-up-'));
	}

	setTimeout(() => {
		document.dispatchEvent(new CustomEvent('entity-drawer:close'));
	}, 300);

	fetchEventSource('/dashboard/user/generator/generate-stream', {
		openWhenHidden: true,
		method: 'POST',
		headers: {
			'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content'),
		},
		body: formData,
		signal: responseObj.abortController.signal,
		onmessage: async e => {
			const txt = e.data;

			if (e.event === 'chat_not_found') {
				// Chat was deleted (e.g. empty chat cleanup) — silently reload to generate a new chat
				window.location.reload();
				return;
			}

			if (e.event === 'clear_content') {
				// Clear streamed AI text when a tool call replaces it (e.g. post generation)
				responseObj.response = [];
				if (responseObj.chatContentEl) {
					responseObj.chatContentEl.innerHTML = '';
				}
				return;
			}

			if (e.event === 'function_call' && e.data === 'search_images') {
				const containerEl = responseObj.chatContentContainerEl || responseObj.bubbleEl?.querySelector('.chat-content-container');
				// Only insert shimmer if one doesn't already exist
				if (containerEl && !containerEl.querySelector('.lqd-smart-images-shimmer')) {
					const shimmerHtml = '<div class="lqd-smart-images-shimmer"><div class="lqd-smart-image-grid grid grid-cols-3 gap-1 rounded-lg overflow-hidden"><div class="lqd-smart-image-shimmer-item relative aspect-[4/3] overflow-hidden rounded bg-foreground/5 lqd-shimmer-effect"></div><div class="lqd-smart-image-shimmer-item relative aspect-[4/3] overflow-hidden rounded bg-foreground/5 lqd-shimmer-effect"></div><div class="lqd-smart-image-shimmer-item relative aspect-[4/3] overflow-hidden rounded bg-foreground/5 lqd-shimmer-effect"></div></div></div>';
					containerEl.insertAdjacentHTML('afterbegin', shimmerHtml);
				}
				return;
			}

			if (e.event === 'smart_images') {
				try {
					const payload = JSON.parse(txt);
					const imagesRaw = payload.images || '';
					// Extract the :::smart-images block - store for persistence at stream end
					const jsonMatch = imagesRaw.match(/:::smart-images\s*\n([\s\S]*?)\n\s*:::/);
					if (jsonMatch) {
						const imagesData = JSON.parse(jsonMatch[1]);
						// Do NOT store for response array — images are persisted to DB by the server.
						// Adding to response causes VDOM re-render conflicts that remove the grid.

						// Build grid directly in DOM (outside VDOM zone) replacing shimmer
						const containerEl = responseObj.chatContentContainerEl || responseObj.bubbleEl?.querySelector('.chat-content-container');
						if (containerEl) {
							const shimmer = containerEl.querySelector('.lqd-smart-images-shimmer');
							const gridEl = buildSmartImageGrid(imagesData);
							if (shimmer) {
								shimmer.replaceWith(gridEl);
							} else {
								containerEl.insertAdjacentElement('afterbegin', gridEl);
							}
						}
					}
				} catch (err) {}
				return;
			}

			// Async parallel image search — fetches images while text streams simultaneously
			if (e.event === 'smart_image_search') {
				try {
					const searchData = JSON.parse(txt);
					if (searchData.query) {
						// Mark that an async image fetch is in progress
						responseObj._asyncImageFetchPending = true;
						const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
						fetch('/dashboard/user/smart-image/search', {
							method: 'POST',
							headers: {
								'Content-Type': 'application/json',
								'X-CSRF-TOKEN': csrfToken,
							},
							body: JSON.stringify({
								query: searchData.query,
								message_id: searchData.message_id,
							}),
						})
							.then(res => res.json())
							.then(data => {
								responseObj._asyncImageFetchPending = false;
								if (data.images && data.images.length > 0) {
									const containerEl = responseObj.chatContentContainerEl || responseObj.bubbleEl?.querySelector('.chat-content-container');
									if (containerEl) {
										const shimmer = containerEl.querySelector('.lqd-smart-images-shimmer');
										const gridEl = buildSmartImageGrid(data.images);
										if (shimmer) {
											shimmer.replaceWith(gridEl);
										} else {
										// Shimmer may have been removed at [DONE] — insert grid at top
											containerEl.insertAdjacentElement('afterbegin', gridEl);
										}
									}
									// Images are already saved to DB by the async endpoint.
									// Do NOT add :::smart-images block to response array — it causes
									// VDOM re-render conflicts that remove the DOM-inserted grid.
									// On page reload, images render from DB via Blade template.
									responseObj._smartImagesBlock = null;
								} else {
								// No images found — remove shimmer
									const containerEl = responseObj.chatContentContainerEl || responseObj.bubbleEl?.querySelector('.chat-content-container');
									if (containerEl) {
										const shimmer = containerEl.querySelector('.lqd-smart-images-shimmer');
										if (shimmer) shimmer.remove();
									}
								}
							})
							.catch(() => {
								responseObj._asyncImageFetchPending = false;
								const containerEl = responseObj.chatContentContainerEl || responseObj.bubbleEl?.querySelector('.chat-content-container');
								if (containerEl) {
									const shimmer = containerEl.querySelector('.lqd-smart-images-shimmer');
									if (shimmer) shimmer.remove();
								}
							});
					}
				} catch (err) {}
				return;
			}

			if (e.event === 'title') {
				try {
					const titleData = JSON.parse(txt);
					if (titleData.title && titleData.chat_id) {
						applyStreamedChatTitle(titleData.chat_id, titleData.title);
						stripTitleFromBubble(responseObj);
						throttledOnAiResponse(responseObj);
					}
				} catch (err) {}
				return;
			}

			if (e.event === 'suggestions') {
				try {
					const suggestionsData = JSON.parse(txt);
					if (suggestionsData?.suggestions?.length) {
						responseObj._streamedSuggestions = suggestionsData.suggestions;
						stripSuggestionsFromBubble(responseObj);
						throttledOnAiResponse(responseObj);
						applySuggestionsToContainer(responseObj);
					}
				} catch (err) {}
				return;
			}

			if (e.event === 'skills_used') {
				try {
					const skillsData = JSON.parse(txt);
					if (skillsData?.skills?.length) {
						responseObj._usedSkills = skillsData.skills;
						throttledOnAiResponse(responseObj);
					}
				} catch (err) {}
				return;
			}

			if (e.event === 'entity_highlights') {
				try {
					const entities = JSON.parse(txt);
					if (Array.isArray(entities) && entities.length > 0) {
						responseObj.entityHighlights = entities;
					}
				} catch (err) {}
				return;
			}

			if (!receivedMessageId) {
				const eventData = e.event
					.split('\n')
					.reduce((acc, line) => {
						if (line.startsWith('message')) {
							acc.type = 'message';
							acc.data = e.data;
						}

						return acc;
					}, {});

				if (eventData.type === 'message') {
					const responseId = eventData.data;

					receivedMessageId = true;

					responseObj.responseId = responseId;

					responseObj.bubbleEl.setAttribute('data-message-id', responseId);
					responseObj.acceptButtonEl?.setAttribute('data-message-id', responseId);
					responseObj.regenerateButtonEl?.setAttribute('data-message-id', responseId);
				}

				return;
			}

			if (txt == null) return;

			const responseIndex = aiResponses.findIndex(response => response.responseId === responseObj.responseId);
			const isDoneSignal = txt.includes('[DONE]');

			if (isDoneSignal) {
				messages.push({
					role: 'assistant',
					content: responseObj.request?.mode === 'council'
						? (responseObj.request?.finalAnswer || getAiResponseString({ responseObj }))
						: getAiResponseString({ responseObj }),
				});

				if (messages.length >= 6) {
					messages.splice(1, 2);
				}

				stripSuggestionsFromBubble(responseObj);

				// Remove shimmer only if no async image fetch is still in progress
				if (responseObj.bubbleEl && !responseObj._asyncImageFetchPending) {
					const containerEl = responseObj.chatContentContainerEl || responseObj.bubbleEl.querySelector('.chat-content-container');
					if (containerEl) {
						const shimmer = containerEl.querySelector('.lqd-smart-images-shimmer');
						if (shimmer) shimmer.remove();
					}
				}

				// Mark stream as done so async image fetch can push to response when it completes
				responseObj._streamDone = true;

				// Add smart images block to response array for persistence (only at stream end)
				// and remove the direct DOM grid since VDOM will now render it inside chat-content
				if (responseObj._smartImagesBlock) {
					responseObj.response.unshift(responseObj._smartImagesBlock);
					delete responseObj._smartImagesBlock;
					// Remove the streaming-time grid from chat-content-container
					// since processSmartImageContainers will create one inside chat-content
					const containerEl2 = responseObj.chatContentContainerEl || responseObj.bubbleEl?.querySelector('.chat-content-container');
					if (containerEl2) {
						const streamGrid = containerEl2.querySelector(':scope > .lqd-smart-image-grid');
						if (streamGrid) streamGrid.remove();
					}
				}

				// done signal should not be pushed to response

				responseObj.responseStreaming = false;
				responseObj.abortController = null;
				if (responseObj.request?.mode === 'council') {
					finalizeCouncilBubbleAnimationState(responseObj);
				} else if (responseObj.bubbleEl) {
					responseObj.bubbleEl.classList.remove('loading', 'animating-words', 'streaming-on');
					responseObj.bubbleEl.classList.add('animating-words-done');
					responseObj.animatingWordIndex = -1;
					switchGenerateButtonsStatus(aiResponses.every(res => res.responseStreaming));
				}

				throttledOnAiResponse(responseObj);

				if (canvasModeActivated) {
					handleCanvasResponseStore(responseObj);
				}

				if (responseIndex === aiResponses.length - 1) {
					window.removeEventListener('beforeunload', onBeforePageUnload);
				}
				document.dispatchEvent(new CustomEvent('chat:response-complete', {
					detail: {
						messageId: responseObj.responseId,
						bubbleEl: responseObj.bubbleEl,
						suggestionsHandled: !!responseObj._streamedSuggestions
					}
				}));

				// Apply entity highlight annotations after words are unwrapped
				if (responseObj.entityHighlights) {
					waitForUnwrapThenHighlight(responseObj);
				}

				return;
			}

			if (responseObj.request?.mode === 'council') {
				try {
					const payload = JSON.parse(txt);
					const shouldReRender = handleCouncilStreamPayload(responseObj, payload);
					if (shouldReRender) {
						throttledOnAiResponse(responseObj);
					}
				} catch (error) {
					responseObj.response.push(txt);
					throttledOnAiResponse(responseObj);
				}
				return;
			}

			// Only append content from data events — ignore any unhandled event types
			// (e.g. title, suggestions, skills_used) so their payloads never leak into the bubble.
			if (e.event && e.event !== 'data' && e.event !== 'message') {
				return;
			}

			responseObj.response.push(txt);

			throttledOnAiResponse(responseObj);
		},
		onerror: err => {
			window.removeEventListener('beforeunload', onBeforePageUnload);

			switchGenerateButtonsStatus(false);

			responseObj.abortController = null;

			responseObj.responseStreaming = false;
			responseObj.response.push(`${magicai_localize.error}: ${err.message}`);

			if (responseObj.request?.mode === 'council') {
				setCouncilSummaryThinking(responseObj, false);
				finalizeCouncilBubbleAnimationState(responseObj);
			}

			throttledOnAiResponse(responseObj);

			throw err;
		},
	});
}

async function startGenerateRequest(ev) {
	'use strict';

	ev?.preventDefault();

	const promptInput = document.getElementById('prompt');
	const promptInputValue = buildPrompt(promptInput.value);

	if (!promptInputValue.trim()) {
		return toastr.error(magicai_localize?.please_fill_message || 'Please fill the message field',);
	}

	const generateBtn = document.querySelector('#send_message_button');
	const chatbotFrontModel = getFrontModelEl();
	const chatsWrapper = document.querySelector('.chats-wrap');
	const chatsContainer = document.querySelector('.chats-container');
	const userBubbleTemplate = document.querySelector('#chat_user_bubble').content.cloneNode(true).firstElementChild;
	const chatType = document.querySelector('#chatType')?.value;
	const mainUpscaleSrc = document.querySelector('#mainupscale_src');
	const suggestions = document.querySelector('#sugg');
	const chat_id = document.querySelector('#chat_id')?.value;
	const selectedModelSlugs = Array.from(chatbotFrontModel.selectedOptions).map(option => option?.value).filter(Boolean);
	const councilModeEnabled = isCouncilModeActive() && (chatType === 'chatPro' || chatType === 'chatpro');
	const multiModelsSelected = !councilModeEnabled && selectedModelSlugs.length > 1;
	const sharedMessageUUID = (multiModelsSelected || councilModeEnabled) ? generateUUID() : null;
	const chatsV2 = Alpine.store('chatsV2');

	if (councilModeEnabled && selectedModelSlugs.length < 2) {
		return toastr.error('Please select at least 2 models for Council Mode.');
	}

	if (councilModeEnabled && chatAttachments.length > 0) {
		return toastr.error('Model Council currently supports text-only prompts.');
	}

	chatsWrapper.classList.remove('conversation-not-started');
	chatsWrapper.classList.add('conversation-started');

	Alpine.store('realtimeChatStatus')?.setConversationStarted(true);

	if (generateBtn.classList.contains('submitting')) return;

	// Clean up any existing animations before resetting
	aiResponses.forEach(responseObj => {
		resetAnimationState(responseObj);
	});

	aiResponses = [];

	window.addEventListener('beforeunload', onBeforePageUnload);

	switchGenerateButtonsStatus(true);

	hideTempNote();

	userBubbleTemplate.querySelector('.chat-content').innerHTML = promptInputValue;

	// Inject highlight-to-ask quote into user bubble
	if ( chatsV2 && chatsV2.selectedTools ) {
		const followUps = chatsV2.selectedTools.filter(tool => tool.startsWith('follow-up-'));
		if (followUps.length) {
			const quoteText = followUps.map(f => f.replace('follow-up-', '')).join(', ');
			const quoteEl = document.createElement('blockquote');
			quoteEl.textContent = quoteText;
			quoteEl.className = 'mb-3 line-clamp-3 w-full border-s border-foreground/10 ps-2 text-2xs italic';
			const chatContent = userBubbleTemplate.querySelector('.chat-content');
			chatContent.insertBefore(quoteEl, chatContent.firstChild);
		}
	}

	// Show selected skill icon in user bubble & clear badges
	const skillInput = document.getElementById('selected_skill_ids');
	// Skills are not compatible with Council Mode — silently clear any leftover selections
	if (councilModeEnabled && skillInput?.value) {
		skillInput.value = '';
	}
	const skillIds = skillInput?.value ? skillInput.value.split(',').filter(Boolean) : [];
	if (skillIds.length > 0) {
		const chatsV2 = Alpine.store('chatsV2');
		const skillNames = [];

		if ( chatsV2 && chatsV2.selectedTools ) {
			chatsV2.selectedTools.forEach(tool => {
				if ( !tool.startsWith('skill-') ) return;

				skillNames.push(tool.split('skill-')[1]);
			});
		}

		if (skillNames.length > 0) {
			const actionsWrap = userBubbleTemplate.querySelector('.lqd-chat-actions-wrap');
			if (actionsWrap) {
				const skillCopyWrap = document.createElement('div');
				skillCopyWrap.className = 'group/skill-btn flex flex-col gap-2 transition-all';
				skillCopyWrap.innerHTML =
					`<button class="group/btn relative inline-flex size-10 items-center justify-center rounded-full border-none bg-white p-0 text-[12px] text-black shadow-lg transition-all hover:-translate-y-[2px] hover:scale-110" title="${skillNames.join(', ')}">` +
						'<span class="pointer-events-none absolute end-full top-1/2 me-1 inline-block -translate-y-1/2 translate-x-1 whitespace-nowrap rounded-full bg-white px-3 py-1 font-medium leading-5 opacity-0 shadow-lg transition-all group-hover/btn:translate-x-0 group-hover/btn:opacity-100">' +
							skillNames.join(', ') +
						'</span>' +
						'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="size-4"><path d="M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5"/></svg>' +
					'</button>';
				actionsWrap.appendChild(skillCopyWrap);
			}
		}
	}

	handlePromptHistory(promptInputValue);

	chatsContainer.insertAdjacentElement('beforeend', userBubbleTemplate);

	document.dispatchEvent(new CustomEvent('chat:navigator-refresh'));

	if (mainUpscaleSrc) {
		mainUpscaleSrc.style.display = 'none';
	}
	if (suggestions) {
		suggestions.style.display = 'none';
	}

	chatAttachments.forEach(({ data, name }) => {
		const chatAttachmentBubbleTemplate = document.querySelector('#chat_user_image_bubble').content.cloneNode(true).firstElementChild;
		const linkElement = chatAttachmentBubbleTemplate.querySelector('a');

		if (data.startsWith('data:image/')) {
			linkElement.href = data;
			chatAttachmentBubbleTemplate.querySelector('svg')?.remove();
			chatAttachmentBubbleTemplate.querySelector('.img-content').src = data;
		} else {
			// For non-image files, create a download link
			linkElement.href = data;
			linkElement.download = name;
			linkElement.target = '_self';

			const fileNameSpan = document.createElement('span');
			fileNameSpan.textContent = name;

			linkElement.insertAdjacentElement('beforeend', fileNameSpan);
			linkElement.removeAttribute('data-fslightbox');
			linkElement.removeAttribute('data-type');
			chatAttachmentBubbleTemplate.querySelector('img')?.remove();
		}

		chatsContainer.insertAdjacentElement('beforeend', chatAttachmentBubbleTemplate);
	});

	throttledRefreshFsLightbox();

	createAiResponses({
		councilMode: councilModeEnabled,
	});

	scrollConversationArea({ smooth: true });

	if (chatAttachments.length == 0) {
		messages.push({
			role: 'user',
			content: promptInputValue,
		});
	} else {
		messages.push({
			role: 'user',
			content: promptInputValue,
		});
	}

	if (category.slug == 'ai_chat_image') {
		let image_formData = new FormData();

		image_formData.append('prompt', promptInputValue);
		image_formData.append('chatHistory', JSON.stringify(messages));

		let response = await $.ajax({
			url: '/dashboard/user/openai/image/generate',
			type: 'POST',
			data: image_formData,
			processData: false,
			contentType: false,
		});

		const chatImageBubbleTemplate = document.querySelector('#chat_bot_image_bubble').content.cloneNode(true).firstElementChild;

		chatImageBubbleTemplate.querySelector('a').href = response.path;
		chatImageBubbleTemplate.querySelector('.img-content').src = response.path;

		chatsContainer.insertAdjacentElement('beforeend', chatImageBubbleTemplate);

		messages.push({
			role: 'assistant',
			content: '',
		});

		if (messages.length >= 6) {
			messages.splice(1, 2);
		}

		saveResponseAsync(promptInputValue, '', chat_id, '', '', '', response.path);

		switchGenerateButtonsStatus(false);

		window.removeEventListener('beforeunload', onBeforePageUnload);

		throttledRefreshFsLightbox();

		scrollConversationArea();

		return;
	}

	pdfName = '';
	pdfPath = '';

	if (chatAttachments.length) {
		let files = [ ...chatAttachments ];

		chatAttachments = [];
		updatePromptFiles();

		$.ajax({
			type: 'POST',
			url: '/files/upload',
			data: {
				files: files,
				_token: document.querySelector('meta[name="csrf-token"]').getAttribute('content'),
			},
			success: result => {
				if (result.type === 'image') {
					aiResponses.forEach(responseObj =>
						sendRequest(
							'vision',
							result.path,
							responseObj,
							sharedMessageUUID
						)
					);
				} else if (result.type === 'other') {
					pdfName = result.name;
					pdfPath = result.path;
					aiResponses.forEach(responseObj =>
						sendRequest(
							'chatPro',
							null,
							responseObj,
							sharedMessageUUID
						)
					);
				}
				promptInput.value = '';
				promptInput.style.height = '';

				promptInput.dispatchEvent(new Event('input', { bubbles: true }));
			},
		});

		return;
	}

	if (councilModeEnabled) {
		const councilResponseObj = aiResponses[0];
		const currentRunId = ++councilRunSequence;
		// Capture prompt before it's cleared below
		const capturedPrompt = promptInput.value;
		councilResponseObj.request = {
			mode: 'council',
			councilModelSlugs: selectedModelSlugs,
			skipModelReplay: true,
			preserveState: true,
			finalAnswer: '',
			capturedPrompt: capturedPrompt,
		};
		prepareCouncilBubbleForStream(councilResponseObj);
		setCouncilSummaryThinking(councilResponseObj, true);

		streamCouncilModelRequests(selectedModelSlugs, sharedMessageUUID, councilResponseObj)
			.then(results => {


				if (currentRunId !== councilRunSequence) {

					return;
				}
				if (!councilResponseObj.responseStreaming) {

					return;
				}


				// Temporarily restore prompt so sendRequest picks it up
				promptInput.value = capturedPrompt;
				sendRequest(
					'chatPro-council-summary',
					null,
					councilResponseObj,
					sharedMessageUUID,
					{
						mode: 'council',
						councilModelSlugs: selectedModelSlugs,
						skipModelReplay: true,
						preserveState: true,
					}
				);
				promptInput.value = '';
				promptInput.dispatchEvent(new Event('input', { bubbles: true }));
			})
			.catch(err => {

				setCouncilSummaryThinking(councilResponseObj, false);
				finalizeCouncilBubbleAnimationState(councilResponseObj);
			});
	} else {
		aiResponses.forEach(responseObj =>
			sendRequest(
				chatType ?? 'chatbot', null,
				responseObj,
				sharedMessageUUID
			)
		);
	}

	promptInput.value = '';
	promptInput.style.height = '';
	promptInput.dispatchEvent(new Event('input', { bubbles: true }));
}

function reduceOnStop() {
	councilRunSequence++;
	abortCouncilChildStreams();

	aiResponses.forEach(responseObj => {
		responseObj.abortController?.abort();
		responseObj.abortController = null;
		responseObj.responseStreaming = false;

		fetch('/dashboard/user/generator/reduce-tokens/chat', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
			},
			body: JSON.stringify({
				streamed_text: getAiResponseString({ responseObj }),
				streamed_message_id: responseObj.responseId
			})
		});
	});
}

function stopGenerateRequest() {
	switchGenerateButtonsStatus(false);

	reduceOnStop();
}

function updateChatButtons() {
	const generateBtn = document.querySelector('#send_message_button');
	const stopBtn = document.querySelector('#stop_button');
	const promptInput = document.querySelector('#prompt');
	const acceptButtonEls = document.querySelectorAll('.multi-model-response-accept');
	const regenerateButtonEls = document.querySelectorAll('.multi-model-response-regenerate');

	generateBtn?.removeEventListener('click', startGenerateRequest);
	stopBtn?.removeEventListener('click', stopGenerateRequest);
	acceptButtonEls.forEach(el => el.removeEventListener('click', onAcceptResponseButtonClick));
	regenerateButtonEls.forEach(el => el.removeEventListener('click', onRegenerateResponseButtonClick));

	if (promptInput) {
		promptInput.addEventListener('keypress', ev => {
			if (ev.code == 'Enter' && !ev.shiftKey) {
				ev.preventDefault();
				$('.lqd-chat-record-trigger').show();
				return startGenerateRequest();
			}
		});
	}

	generateBtn?.addEventListener('click', startGenerateRequest);
	stopBtn?.addEventListener('click', stopGenerateRequest);
	acceptButtonEls.forEach(el => el.addEventListener('click', onAcceptResponseButtonClick));
	regenerateButtonEls.forEach(el => el.addEventListener('click', onRegenerateResponseButtonClick));
}

function updateFav(id) {
	$.ajax({
		type: 'post',
		url: '/dashboard/user/openai/chat/update-prompt',
		data: {
			id: id,
		},
		success: function (data) {
			favData = data;
			updatePrompts(promptsData);
		},
		error: function () {
		},
	});
}

function updatePrompts(data) {
	const $prompts = $('#prompts');

	$prompts.empty();

	if (data.length == 0) {
		$('#no_prompt').removeClass('hidden');
	} else {
		$('#no_prompt').addClass('hidden');
	}

	for (let i = 0; i < data.length; i++) {
		let isFav = favData.filter(item => item.item_id == data[i].id).length;

		let title = data[i].title.toLowerCase();
		let prompt = data[i].prompt.toLowerCase();
		let searchStr = searchString.toLowerCase();

		if (data[i].id == selectedPrompt) {
			if (title.includes(searchStr) || prompt.includes(searchStr)) {
				if ((filterType == 'fav' && isFav != 0) || filterType != 'fav') {
					let prompt = document.querySelector('#selected_prompt').content.cloneNode(true);
					const favbtn = prompt.querySelector('.favbtn');
					prompt.querySelector('.prompt_title').innerHTML = data[i].title;
					prompt.querySelector('.prompt_text').innerHTML = data[i].prompt;
					favbtn.setAttribute('id', data[i].id);

					if (isFav != 0) {
						favbtn.classList.add('active');
					} else {
						favbtn.classList.remove('active');
					}

					$prompts.append(prompt);
				} else {
					selectedPrompt = -1;
				}
			} else {
				selectedPrompt = -1;
			}
		} else {
			if (title.includes(searchStr) || prompt.includes(searchStr)) {
				if (
					(filterType == 'fav' && isFav != 0) ||
					filterType != 'fav'
				) {
					let prompt = document
						.querySelector('#unselected_prompt')
						.content.cloneNode(true);
					const favbtn = prompt.querySelector('.favbtn');
					prompt.querySelector('.prompt_title').innerHTML =
						data[i].title;
					prompt.querySelector('.prompt_text').innerHTML =
						data[i].prompt;
					favbtn.setAttribute('id', data[i].id);

					if (isFav != 0) {
						favbtn.classList.add('active');
					} else {
						favbtn.classList.remove('active');
					}

					$prompts.append(prompt);
				}
			}
		}
	}
	let favCnt = favData.length;
	let perCnt = data.length;

	if (favCnt == 0) {
		$('#fav_count')[0].innerHTML = '';
	} else {
		$('#fav_count')[0].innerHTML = favCnt;
	}

	if (perCnt == 0 || perCnt == undefined) {
		$('#per_count')[0].innerHTML = '';
	} else {
		$('#per_count')[0].innerHTML = perCnt;
	}
}

function searchStringChange(e) {
	searchString = $('#search_str').val();
	updatePrompts(promptsData);
}

function openNewImageDlg(e) {
	$('#selectImageInput').click();
}

function isAllowedFileType(data, name) {
	const allowedFileTypes = {
		image: [ 'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp' ],
		document: [
			'application/pdf',
			'application/msword',
			'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
			'application/vnd.ms-excel',
			'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
			'text/plain'
		]
	};

	const mimeExtensions = {
		'.png': 'image/png',
		'.jpg': 'image/jpeg',
		'.jpeg': 'image/jpeg',
		'.gif': 'image/gif',
		'.webp': 'image/webp',
		'.pdf': 'application/pdf',
		'.doc': 'application/msword',
		'.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
		'.xls': 'application/vnd.ms-excel',
		'.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
		'.txt': 'text/plain'
	};

	let fileType = null;
	const mimeMatch = data.match(/^data:([^;]+);/);
	if (mimeMatch) {
		fileType = mimeMatch[1];
	} else {
		const extMatch = name.match(/\.(\w+)$/);
		if (!extMatch) return false;
		const ext = '.' + extMatch[1].toLowerCase();
		fileType = mimeExtensions[ext] || null;
	}

	return fileType && Object.values(allowedFileTypes).some(arr => arr.includes(fileType));
}

function updatePromptFiles() {
	$('#chat-attachment-previews').empty();

	if (chatAttachments.length == 0) {
		$('#chat-attachment-previews').removeClass('active');
		$('.split_line').addClass('hidden');
		return;
	}

	$('#chat-attachment-previews').addClass('active');
	$('.split_line').removeClass('hidden');

	chatAttachments.forEach(({ data, name }, index) => {

		if (data.startsWith('data:image/')) {
			let newImage = document.querySelector('#prompt_image').content.cloneNode(true).firstElementChild;

			newImage.querySelector('img').setAttribute('src', data);
			newImage.querySelector('.prompt_image_close').setAttribute('index', index);

			document.querySelector('#chat-attachment-previews').insertAdjacentElement('beforeend', newImage);
		} else {
			let newFile = document.querySelector('#prompt_pdf').content.cloneNode(true).firstElementChild;
			const linkElement = newFile.querySelector('a');

			linkElement.href = data;
			linkElement.download = name;
			linkElement.target = '_self';

			newFile.querySelector('a span').textContent = name;
			newFile.querySelector('.prompt_image_close').setAttribute('index', index);

			document.querySelector('#chat-attachment-previews').insertAdjacentElement('beforeend', newFile);
		}
	});

	let new_image_btn = document.querySelector('#prompt_image_add_btn').content.cloneNode(true);

	document.querySelector('#chat-attachment-previews').append(new_image_btn);

	$('.promt_image_btn').on('click', function (e) {
		e.preventDefault();
		$('#chat_add_image').click();
	});

	$('.prompt_image_close').on('click', function () {
		chatAttachments.splice($(this).attr('index'), 1);
		updatePromptFiles();
	});
}

function addFileToChat({ data, name }) {
	if (chatAttachments.find(attachment => attachment.data === data)) return;

	if (!isAllowedFileType(data, name)) {
		console.warn(`File "${name}" is not allowed.`);
		return toastr.error('File is not supported.');
	}

	chatAttachments.push({ data, name });
	updatePromptFiles();
}

function initChat() {
	var mediaRecorder;
	var chunks = [];
	var stream_;

	chatAttachments = [];

	$('#scrollable_content').animate({ scrollTop: 1000 }, 200);

	// Start recording when the button is pressed
	$('#voice_record_button').click(function () {
		chunks = [];
		navigator.mediaDevices
			.getUserMedia({ audio: true })
			.then(function (stream) {
				stream_ = stream;
				mediaRecorder = new MediaRecorder(stream);
				$('#voice_record_button').addClass('inactive');
				$('#voice_record_stop_button').addClass('active');
				mediaRecorder.ondataavailable = function (e) {
					chunks.push(e.data);
				};
				mediaRecorder.start();
			})
			.catch(function (err) {
				console.log('The following error occurred: ' + err);
				toastr.warning('Audio is not allowed');
			});

		$('#voice_record_stop_button').click(function (e) {
			e.preventDefault();
			$('#voice_record_button').removeClass('inactive');
			$('#voice_record_stop_button').removeClass('active');
			mediaRecorder.onstop = function () {
				var blob = new Blob(chunks, { type: 'audio/mp3' });

				var formData = new FormData();
				var fileOfBlob = new File([ blob ], 'audio.mp3');
				formData.append('file', fileOfBlob);

				chunks = [];

				$.ajax({
					url: '/dashboard/user/openai/chat/transaudio',
					type: 'POST',
					data: formData,
					contentType: false,
					processData: false,
					success: function (response) {
						if (response.length >= 4) {
							$('#prompt').val(response);
						}
					},
					error: function () {
						// Handle the error response
					},
				});
			};
			mediaRecorder.stop();
			stream_
				.getTracks() // get all tracks from the MediaStream
				.forEach(track => track.stop()); // stop each of them
		});
	});

	$('#btn_add_new_prompt').on('click', function (e) {
		prompt_title = $('#new_prompt_title').val();
		prompt = $('#new_prompt').val();

		if (prompt_title.trim() == '') {
			toastr.warning('Please input title');
			return;
		}

		if (prompt.trim() == '') {
			toastr.warning('Please input prompt');
			return;
		}

		$.ajax({
			type: 'post',
			url: '/dashboard/user/openai/chat/add-prompt',
			data: {
				title: prompt_title,
				prompt: prompt,
			},
			success: function (data) {
				promptsData = data;
				updatePrompts(data);
				$('.custom__popover__back').addClass('hidden');
				$('#custom__popover').removeClass('custom__popover__wrapper');
			},
			error: function () {
			},
		});
	});

	$('#add_btn').on('click', function (e) {
		$('#custom__popover').addClass('custom__popover__wrapper');
		$('.custom__popover__back').removeClass('hidden');
		e.stopPropagation();
	});

	$('.custom__popover__back').on('click', function () {
		$(this).addClass('hidden');
		$('#custom__popover').removeClass('custom__popover__wrapper');
	});

	$('#prompt_library').on('click', function (e) {
		e.preventDefault();

		$('#prompts').empty();

		$.ajax({
			type: 'post',
			url: '/dashboard/user/openai/chat/prompts',
			success: function (data) {
				filterType = 'all';
				promptsData = data.promptData;
				favData = data.favData;
				updatePrompts(data.promptData);
				$('#modal').addClass('lqd-is-active');
				$('.modal__back').removeClass('hidden');
			},
			error: function () {
			},
		});
		e.stopPropagation();
	});

	$('.modal__back').on('click', function () {
		$(this).addClass('hidden');
		$('#modal').removeClass('lqd-is-active');
	});

	$(document).on('click', '.prompt', function () {
		const $promptInput = $('#prompt');
		selectedPrompt = Number($(this.querySelector('.favbtn')).attr('id'));
		$promptInput.val(
			promptsData.filter(item => item.id == selectedPrompt)[0].prompt,
		);
		$('.modal__back').addClass('hidden');
		$('#modal').removeClass('lqd-is-active');
		selectedPrompt = -1;
		$promptInput.css('height', '5px');
		$promptInput.css('height', $promptInput[0].scrollHeight + 'px');
	});

	$(document).on('click', '.filter_btn', function () {
		$('.filter_btn').removeClass('active');
		$(this).addClass('active');
		filterType = $(this).attr('filter');
		updatePrompts(promptsData);
	});

	$(document).on('click', '.favbtn', function (e) {
		updateFav(Number($(this).attr('id')));
		e.stopPropagation();
	});

	$(document).on('click', '#chat_add_image', () => {
		document.querySelector('#selectImageInput')?.click();
	});

	$('#selectImageInput').change(function () {
		this.files.forEach(file => {
			let reader = new FileReader();

			reader.onload = function (e) {
				addFileToChat({ data: e.target.result, name: file.name });
			};

			reader.readAsDataURL(file);
		});

		if (document.getElementById('mainupscale_src')) {
			document.getElementById('mainupscale_src').style.display = 'none';
		}
	});

	$('#upscale_src').change(function () {
		this.files.forEach(file => {
			let reader = new FileReader();

			reader.onload = function (e) {
				addFileToChat({ data: e.target.result, name: file.name });
			};

			reader.readAsDataURL(file);
		});

		if (document.getElementById('mainupscale_src')) {
			document.getElementById('mainupscale_src').style.display = 'none';
		}
	});

	document
		.querySelectorAll('.lqd-chat-ai-bubble')
		.forEach(aiChatBubble => {
			const contentEl = aiChatBubble.querySelector('.chat-content');

			if ( !contentEl ) return;

			contentEl.classList.remove('!whitespace-pre-wrap', 'whitespace-pre-wrap');
			contentEl.style.whiteSpace = 'normal';

			if (contentEl.classList.contains('is-html')) {
				const turndownService = new TurndownService();
				const contentClone = contentEl.cloneNode(true);
				contentClone
					.querySelectorAll('.lqd-chat-actions-wrap, .lqd-clipboard-copy-wrap, .lqd-clipboard-copy, .lqd-chat-bubble-canvas-trigger, [data-copy-options]')
					.forEach(el => el.remove());
				const markdown = turndownService.turndown(contentClone);

				contentEl.innerHTML = markdown;
			}

			contentEl.innerHTML = formatString(contentEl.innerHTML);

			// Process smart image containers in loaded history messages
			processSmartImageContainers(contentEl);

			// Reveal skill cards that were hidden to prevent raw markdown flash
			if (contentEl.classList.contains('lqd-has-skill-block')) {
				contentEl.classList.remove('opacity-0', 'lqd-has-skill-block');
			}

			throttledRefreshFsLightbox();

			if ( contentEl.querySelector('.social-media-agent-chat-post-card') ) {
				aiChatBubble.querySelector('.lqd-chat-bubble-canvas-trigger')?.remove();
				aiChatBubble.querySelectorAll('[data-copy-options],[data-copy-type]').forEach(el => el.remove());
			}

			requestAnimationFrame(() => {
				const councilReplieContents = aiChatBubble.querySelectorAll('.model-council-reply-content');
				councilReplieContents.forEach(councilReplyContent => councilReplyContent.innerHTML = formatString(councilReplyContent.innerHTML.trim()) );

				// Apply entity highlights scoped to this bubble only
				// Apply entity highlights scoped to this bubble only
				if (aiChatBubble.dataset.entityHighlights) {
					try {
						const entities = JSON.parse(aiChatBubble.dataset.entityHighlights);
						if (entities?.length) {
							applyEntityHighlights(aiChatBubble, entities);
						}
					} catch (e) {
						// Silent fail
					}
				}
			});
		});

	// Refresh fslightbox once after ALL bubbles are processed so smart image links get onclick handlers
	if ('refreshFsLightbox' in window) {
		refreshFsLightbox();
	}
}

async function saveResponseAsync(input, response, chat_id, imagePath, pdfName, pdfPath, outputImage = '', model = '',) {
	var formData = new FormData();

	if (!response) {
		response = '';
	}

	formData.append('chat_id', chat_id);
	formData.append('input', input);
	formData.append('response', response);
	formData.append('images', imagePath);
	formData.append('pdfName', pdfName);
	formData.append('pdfPath', pdfPath);
	formData.append('outputImage', outputImage);
	formData.append('model', model);

	try {
		const result = await jQuery.ajax({
			url: '/dashboard/user/openai/chat/low/chat_save',
			type: 'POST',
			headers: {
				'X-CSRF-TOKEN': '{{ csrf_token() }}',
			},
			data: formData,
			contentType: false,
			processData: false,
		});
		if (result.status === 'error') {
			toastr.error(result.message, 'Error');
		}

		return result;
	} catch (error) {
		if (error.responseJSON && error.responseJSON.message) {
			toastr.error(error.responseJSON.message, 'Error');
		} else {
			toastr.error('An unexpected error occurred. Please try again.', 'Error');
		}
	}
	return false;
}

/*

DO NOT FORGET TO ADD THE CHANGES TO BOTH FUNCTION makeDocumentReadyAgain and the document ready function on the top!!!!

*/
function makeDocumentReadyAgain() {
	const chatsWrapper = document.querySelector('.chats-wrap');
	const chatBubbles = chatsWrapper?.querySelectorAll('.lqd-chat-ai-bubble, .lqd-chat-user-bubble');

	_.defer(() => {
		setChatsCssVars();
		updateChatButtons();
	});

	$(document).ready(function () {
		'use strict';

		const chat_id = $('#chat_id').val();
		$(`#chat_${chat_id}`)
			.addClass('active')
			.siblings()
			.removeClass('active');

		scrollConversationArea();

		handlePromptHistoryNavigate();
	});

	if (chatBubbles) {
		chatsWrapper.classList.toggle('conversation-not-started', chatBubbles.length <= 1);
		chatsWrapper.classList.toggle('conversation-started', chatBubbles.length > 1);
	}
}

function handlePromptHistory(prompt) {
	const promptHistory = localStorage.getItem('promptHistory');

	if (!promptHistory) {
		return localStorage.setItem('promptHistory', JSON.stringify([ prompt ]));
	}

	const promptHistoryArray = JSON.parse(promptHistory);

	if (promptHistoryArray.includes(prompt)) {
		promptHistoryArray.splice(promptHistoryArray.indexOf(prompt), 1);
	}

	promptHistoryArray.push(prompt);

	localStorage.setItem('promptHistory', JSON.stringify(promptHistoryArray));
}

function removePromptHistoryHandler() {
	const promptInput = document.querySelector('.lqd-chat-form #prompt');

	promptInput?.removeEventListener('keydown', onPromptInputKeyUpDown);
}

function handlePromptHistoryNavigate() {
	const promptInput = document.querySelector('.lqd-chat-form #prompt');

	if (!promptInput) return;

	promptInput.addEventListener('keydown', onPromptInputKeyUpDown);
}

function onPromptInputKeyUpDown(e) {
	const promptInput = e.target;
	const promptHistory = localStorage.getItem('promptHistory') || '[]';
	const promptHistoryArray = JSON.parse(promptHistory);

	if (promptHistoryArray.length === 0) return;

	if (promptInput.value !== '' && !navigatingInChatsHistory) {
		return;
	}

	const arrowsPressed = e.key === 'ArrowUp' || e.key === 'ArrowDown';

	if (e.key === 'ArrowUp') {
		navigatingInChatsHistory = true;

		if (selectedHistoryPrompt === -1) {
			selectedHistoryPrompt = promptHistoryArray.length - 1;
		} else {
			selectedHistoryPrompt = Math.max(0, selectedHistoryPrompt - 1);
		}

		promptInput.value = promptHistoryArray[selectedHistoryPrompt];
	}

	if (e.key === 'ArrowDown') {
		navigatingInChatsHistory = true;

		if (selectedHistoryPrompt === -1) {
			selectedHistoryPrompt = 0;
		} else {
			selectedHistoryPrompt = Math.min(
				promptHistoryArray.length - 1,
				selectedHistoryPrompt + 1,
			);
		}

		promptInput.value = promptHistoryArray[selectedHistoryPrompt];
	}

	if (!arrowsPressed) {
		navigatingInChatsHistory = false;
		selectedHistoryPrompt = -1;
	}
}

handlePromptHistoryNavigate();

function escapeHtml(html) {
	var text = document.createTextNode(html);
	var div = document.createElement('div');
	div.appendChild(text);
	return div.innerHTML;
}

function openChatAreaContainer(chat_id, website_url = null) {

	chatid = chat_id;
	$(`#chat_${chat_id}`).addClass('active').siblings().removeClass('active');

	var formData = new FormData();

	formData.append('chat_id', chat_id);

	if (website_url != null && website_url != '') {
		formData.append('website_url', website_url);
	}

	let openChatAreaContainerUrl = $('#openChatAreaContainerUrl').val();

	return $.ajax({
		type: 'post',
		url: openChatAreaContainerUrl,
		data: formData,
		contentType: false,
		processData: false,
		success: function (data) {
			removePromptHistoryHandler();

			// Clean up highlight-to-ask and entity drawer state from previous chat
			document.dispatchEvent(new CustomEvent('entity-drawer:close'));
			document.dispatchEvent(new CustomEvent('highlight-to-ask:cleanup'));

			$('#load_chat_area_container > .lqd-card-body').html(data.html);

			initChat();

			messages = [
				{
					role: 'assistant',
					content: prompt_prefix,
				},
			];

			data.lastThreeMessage.forEach(message => {
				messages.push({
					role: 'user',
					content: message.input,
				});
				messages.push({
					role: 'assistant',
					content: message.output,
				});
			});

			makeDocumentReadyAgain();
			if (data.lastThreeMessage != '') {
				if (document.getElementById('mainupscale_src')) {
					document.getElementById('mainupscale_src').style.display = 'none';
				}
				if (document.getElementById('sugg')) {
					document.getElementById('sugg').style.display = 'none';
				}
			}
			setTimeout(function () {
				scrollConversationArea();
				document.dispatchEvent(new CustomEvent('chat:navigator-refresh'));
			}, 750);

			// Re-apply entity highlights to the newly loaded messages
			requestAnimationFrame(function () {
				document.querySelectorAll('[data-entity-highlights]').forEach(function (messageEl) {
					try {
						var entities = JSON.parse(messageEl.dataset.entityHighlights);
						if (entities && entities.length) {
							applyEntityHighlights(messageEl, entities);
						}
					} catch (e) {
						// Silent fail
					}
				});
			});
		},
		error: function (data) {
			var err = data.responseJSON.errors;
			if (err) {
				$.each(err, function (index, value) {
					toastr.error(value);
				});
			} else {
				toastr.error(data.responseJSON.message);
			}
		},
	});
}

function startNewChat(category_id, local, website_url = null) {
	const formData = new FormData();
	const chatsWrapper = document.querySelector('.chats-wrap');
	formData.append('category_id', category_id);

	// let website_url = $("#website_url")?.val();
	let createChatUrl = $('#createChatUrl')?.val();

	if (website_url != null && website_url != '') {
		formData.append('website_url', website_url);
	}

	let link = '/dashboard/user/openai/chat/start-new-chat';

	if (createChatUrl) {
		link = createChatUrl;
	}

	return $.ajax({
		type: 'post',
		url: link,
		data: formData,
		contentType: false,
		processData: false,
		success: function (data) {
			removePromptHistoryHandler();

			chatid = data.chat.id;

			chatsWrapper.classList.remove('conversation-started');
			chatsWrapper.classList.add('conversation-not-started');

			$('#load_chat_area_container > .lqd-card-body').html(data.html);

			document.dispatchEvent(new CustomEvent('chat-sidebar:refresh'));
			document.dispatchEvent(new CustomEvent('chat-sidebar:set-active', { detail: { chatId: data.chat.id } }));

			initChat();

			messages = [
				{
					role: 'assistant',
					content: prompt_prefix,
				},
			];

			makeDocumentReadyAgain();

			setTimeout(function () {
				scrollConversationArea();
				document.dispatchEvent(new CustomEvent('chat:navigator-refresh'));
			}, 750);

			setTimeout(() => {
				const promptEl = document.querySelector('#prompt');

				if ( promptEl ) {
					promptEl.value = '';
					promptEl.dispatchEvent(new Event('input', { bubbles: true }));
				}
			}, 0);

			document.dispatchEvent(new CustomEvent('chat-created', { bubbles: true, detail: { chatId: data.chat.id } }));
		},
		error: function (data) {
			var err = data.responseJSON.errors;
			if (err) {
				$.each(err, function (index, value) {
					toastr.error(value);
				});
			} else {
				toastr.error(data.responseJSON.message);
			}
		},
	});
}

function deleteAllConv(category_id) {
	if (confirm('Are you sure you want to remove all chats?')) {
		if (category_id == 0) {
			toastr.error('Please select a category');
			return false;
		}

		var formData = new FormData();
		const searchInput = document.querySelector('#chat_search_word');
		const website_url = searchInput ? searchInput.getAttribute('data-website-url') : null;
		formData.append('category_id', category_id);
		if (website_url != null && website_url != '') {
			formData.append('website_url', website_url);
		}
		let link = '/dashboard/user/openai/chat/clear-chats';
		$.ajax({
			type: 'post',
			url: link,
			data: formData,
			contentType: false,
			processData: false,
			success: function (data) {
				document.dispatchEvent(new CustomEvent('chat-sidebar:refresh'));
				toastr.success(data.message || 'All chats cleared successfully.');
			},
			error: function (data) {
				var err = data.responseJSON.errors;
				if (err) {
					$.each(err, function (index, value) {
						toastr.error(value);
					});
				} else {
					toastr.error(data.responseJSON.message);
				}
			},
		});
		return false;
	}
}

function startNewDocChat(file, type) {
	'use strict';

	let category_id = $('#chat_search_word').data('category-id');

	var formData = new FormData();
	formData.append('category_id', category_id);
	formData.append('doc', pdf);
	formData.append('type', type);

	Alpine.store('appLoadingIndicator').show();
	$('.lqd-upload-doc-trigger').attr('disabled', true);

	$.ajax({
		type: 'post',
		url: '/dashboard/user/openai/chat/start-new-doc-chat',
		data: formData,
		contentType: false,
		processData: false,
		success: function (data) {
			removePromptHistoryHandler();
			Alpine.store('appLoadingIndicator').hide();
			$('.lqd-upload-doc-trigger').attr('disabled', false);
			$('#selectDocInput').val('');
			chatid = data.chat.id;
			$('#load_chat_area_container > .lqd-card-body').html(data.html);

			document.dispatchEvent(new CustomEvent('chat-sidebar:refresh'));
			document.dispatchEvent(new CustomEvent('chat-sidebar:set-active', { detail: { chatId: data.chat.id } }));

			initChat();
			messages = [
				{
					role: 'assistant',
					content: prompt_prefix,
				},
			];
			makeDocumentReadyAgain();
			setTimeout(function () {
				$('.conversation-area')
					.stop()
					.animate(
						{ scrollTop: $('.conversation-area').outerHeight() },
						200,
					);
			}, 750);

			toastr.success(magicai_localize.analyze_file_finish);
		},
		error: function (data) {
			Alpine.store('appLoadingIndicator').hide();
			$('.lqd-upload-doc-trigger').attr('disabled', false);
			$('#selectDocInput').val('');
			var err = data.responseJSON.errors;
			if (err) {
				$.each(err, function (index, value) {
					toastr.error(value);
				});
			} else {
				toastr.error(data.responseJSON.message);
			}
		},
	});
	return false;
}

function searchChatFunction() {
	'use strict';

	const input = document.querySelector('#chat_search_word');
	const categoryId = input.getAttribute('data-category-id');
	const website_url = input.getAttribute('data-website-url');

	const formData = new FormData();

	formData.append('_token', document.querySelector('input[name=_token]')?.value);
	formData.append('search_word', input.value);
	formData.append('category_id', categoryId);

	if ( website_url && website_url != null ) {
		formData.append('website_url', website_url);
	}

	$.ajax({
		type: 'POST',
		url: '/dashboard/user/openai/chat/search',
		data: formData,
		contentType: false,
		processData: false,
		success: function (result) {
			$('#chat_sidebar_container').html(result.html);
			$(document).trigger('ready');
		},
	});
}

/**
 * @param {object} opts
 * @param {'end' | number} opts.y
 * @param {boolean} opts.smooth
 */
function scrollConversationArea(opts = {}) {
	const options = {
		y: 'end',
		smooth: false,
		...opts,
	};
	const el = document.querySelector('.conversation-area');

	if (!el) return;

	const y = options.y === 'end' ? el.scrollHeight + 200 : options.y;

	el.scrollTo({
		top: Math.round(y),
		left: 0,
		behavior: options.smooth ? 'smooth' : 'auto',
	});
}

function saveResponse(input, response, chat_id, imagePath = '', pdfName = '', pdfPath = '', outputImage = '') {
	var formData = new FormData();
	formData.append('chat_id', chat_id);
	formData.append('input', input);
	formData.append('response', response);
	formData.append('images', imagePath);
	formData.append('pdfName', pdfName);
	formData.append('pdfPath', pdfPath);
	formData.append('outputImage', outputImage);
	jQuery.ajax({
		url: '/dashboard/user/openai/chat/low/chat_save',
		type: 'POST',
		headers: {
			'X-CSRF-TOKEN': '{{ csrf_token() }}',
		},
		data: formData,
		contentType: false,
		processData: false,
	});
	return false;
}

function addText(text) {
	var promptElement = document.getElementById('prompt');
	var currentText = buildPrompt(promptElement.value);
	var newText = currentText + text;
	promptElement.value = newText;
}

function dropHandler(ev, id) {
	ev.preventDefault();
	const input = document.querySelector(`#${id}`);
	const fileNameEl =
		input?.previousElementSibling?.querySelector('.file-name');

	if (!input) return;

	input.files = ev.dataTransfer.files;

	if (fileNameEl) {
		fileNameEl.innerText = ev.dataTransfer.files[0].name;
	}

	ev.dataTransfer.files.forEach(file => {
		let reader = new FileReader();

		reader.onload = function (e) {
			addFileToChat({ data: e.target.result, name: file.name });
		};

		reader.readAsDataURL(file);
	});

	if (document.getElementById('mainupscale_src')) {
		document.getElementById('mainupscale_src').style.display = 'none';
	}
}

function dragOverHandler(ev) {
	// Prevent default behavior (Prevent file from being opened)
	ev.preventDefault();
}

function handleFileSelect(id) {
	$('#' + id)
		.prev()
		.find('.file-name')
		.text($('#' + id)[0].files[0].name);
}

function exportAsPdf() {
	var win = window.open(
		`/dashboard/user/openai/chat/generate-pdf?id=${chatid}`,
		'_blank',
	);
	win.focus();
}

function exportAsWord() {
	var win = window.open(
		`/dashboard/user/openai/chat/generate-word?id=${chatid}`,
		'_blank',
	);
	win.focus();
}

function exportAsTxt() {
	var win = window.open(
		`/dashboard/user/openai/chat/generate-txt?id=${chatid}`,
		'_blank',
	);
	win.focus();
}

$(document).ready(function () {
	'use strict';

	initChat();

	scrollConversationArea();

	_.defer(updateChatButtons);

	function saveChatNewTitle(chatId, newTitle) {
		var formData = new FormData();
		formData.append('chat_id', chatId);
		formData.append('title', newTitle);

		$.ajax({
			type: 'post',
			url: '/dashboard/user/openai/chat/rename-chat',
			data: formData,
			contentType: false,
			processData: false,
		});
		return false;
	}

	function deleteChatItem(chatId, chatTitle) {
		if (confirm(`Are you sure you want to remove ${chatTitle}?`)) {
			var formData = new FormData();
			formData.append('chat_id', chatId);

			const chatTrigger = $(`#${chatId}`);
			const chatIsActive = chatTrigger.hasClass('active');
			let nextChatToActivate = chatTrigger.prevAll(':visible').first();
			const chatsContainer = document.querySelector('.chats-container');

			if (nextChatToActivate.length === 0) {
				nextChatToActivate = chatTrigger.nextAll(':visible').first();
			}

			$.ajax({
				type: 'post',
				url: '/dashboard/user/openai/chat/delete-chat',
				data: formData,
				contentType: false,
				processData: false,
				success: function (data) {
					//Remove chat li
					chatTrigger.hide();
					if (chatIsActive) {
						if (chatsContainer) {
							chatsContainer.innerHTML = '';
						}
						nextChatToActivate
							.children('.chat-list-item-trigger')
							.click();
					}
					toastr.success(
						magicai_localize.conversation_deleted_successfully,
					);
				},
				error: function (data) {
					var err = data.responseJSON.errors;
					if (err) {
						$.each(err, function (index, value) {
							toastr.error(value);
						});
					} else {
						toastr.error(data.responseJSON.message);
					}
				},
			});
			return false;
		}
	}

	$('#chat_sidebar_container').on('click', '.chat-item-delete', ev => {
		const button = ev.currentTarget;
		const parent = button.closest('li');
		const chatId = parent.getAttribute('id');
		const chatTitle = parent.querySelector('.chat-item-title').innerText;
		deleteChatItem(chatId, chatTitle);
	});

	$('#chat_sidebar_container').on('click', '.chat-item-update-title', ev => {
		const button = ev.currentTarget;
		const parent = button.closest('.chat-list-item');
		const title = parent.querySelector('.chat-item-title');
		const chatId = parent.getAttribute('id');
		const currentText = title.innerText;

		function setEditMode(mode) {
			if (mode === 'editStart') {
				parent.classList.add('edit-mode');

				title.setAttribute('data-current-text', currentText);
				title.setAttribute('contentEditable', true);
				title.focus();
				window.getSelection().selectAllChildren(title);
			} else if (mode === 'editEnd') {
				parent.classList.remove('edit-mode');

				title.removeAttribute('contentEditable');
				title.removeAttribute('data-current-text');
			}
		}

		function keydownHandler(ev) {
			const { key } = ev;
			const escapePressed = key === 'Escape';
			const enterPressed = key === 'Enter';

			if (!escapePressed && !enterPressed) return;

			ev.preventDefault();

			if (escapePressed) {
				title.innerText = currentText;
			}

			if (enterPressed) {
				saveChatNewTitle(chatId, title.innerText);
			}

			setEditMode('editEnd');
			document.removeEventListener('keydown', keydownHandler);
		}

		// if alreay editting then turn the edit button to a save button
		if (title.hasAttribute('contentEditable')) {
			setEditMode('editEnd');
			document.removeEventListener('keydown', keydownHandler);
			return saveChatNewTitle(chatId, title.innerText);
		}

		$('.chat-list-ul .edit-mode').each((i, el) => {
			const title = el.querySelector('.chat-item-title');
			title.innerText = title.getAttribute('data-current-text');
			title.removeAttribute('data-current-text');
			title.removeAttribute('contentEditable');
			el.classList.remove('edit-mode');
		});

		setEditMode('editStart');

		document.addEventListener('keydown', keydownHandler);
	});

	$('#chat_sidebar_container').on('click', '.chat-item-pin', ev => {
		const button = ev.currentTarget;
		const parent = button.closest('.chat-list-item');
		const chatId = parent.getAttribute('id');
		const isPinned = parent.classList.contains('pin-mode');

		function togglePinMode() {
			parent.classList.toggle('pin-mode');
			const pinIcon = button.querySelector('.tabler-pin');
			const pinnedIcon = button.querySelector('.tabler-pinned');
			if (pinIcon && pinnedIcon) {
				pinIcon.classList.toggle('hidden');
				pinnedIcon.classList.toggle('hidden');
			}
		}

		togglePinMode();

		$.ajax({
			type: 'post',
			url: '/dashboard/user/openai/chat/pin-conversation',
			data: JSON.stringify({ pinned: !isPinned, chat_id: chatId }),
			contentType: 'application/json',
			success: function (data) {
				toastr.success(isPinned ? magicai_localize.conversation_unpinned : magicai_localize.conversation_pinned);
			},
			error: (xhr, status, error) => {
				console.error('Error updating pin status:', error);
				togglePinMode();
				toastr.error(magicai_localize.conversation_pin_error);
			},
		});
	});

	$('#chat_search_word').on('keyup', function () {
		return searchChatFunction();
	});

	$('body').on('input', '#prompt', ev => {
		const el = ev.target;
		if (!el.dataset.initialHeight) {
			el.dataset.initialHeight = el.offsetHeight;
		}
		const minHeight = parseInt(el.dataset.initialHeight, 10);
		el.style.height = '5px';
		el.style.height = Math.max(el.scrollHeight, minHeight) + 'px';
		const recordTrigger = $('.lqd-chat-record-trigger');
		const chatsWrapper = $('.chats-wrap');

		// check if value is not empty and then hide .lqd-chat-record-trigger and .lqd-chat-record-stop-trigger elements
		if (
			el.value &&
			el.value !== '' &&
			!(Array.isArray(el.value) && el.value.length === 0) &&
			!(
				typeof el.value === 'object' &&
				Object.keys(el.value).length === 0
			)
		) {
			recordTrigger.hide();
			chatsWrapper.addClass('prompt-filled');
		} else {
			recordTrigger.show();
			chatsWrapper.removeClass('prompt-filled');
		}
	});

	$('#selectDocInput').change(function () {
		if (this.files && this.files[0]) {
			let reader = new FileReader();
			pdf = this.files[0];

			toastr.success(magicai_localize.analyze_file_begin);

			startNewDocChat(pdf, this.files[0].type);

			if (document.getElementById('mainupscale_src')) {
				document.getElementById('mainupscale_src').style.display = 'none';
			}
		}
	});

	window.addEventListener('beforeunload', function (e) {
		reduceOnStop();
	});
});

$('body').on('click', '.chat-download', event => {
	const button = event.currentTarget;
	const docType = button.dataset.docType;
	const docName = button.dataset.docName || 'document';

	const container = document.querySelector('.chats-container');
	let content = container?.parentElement?.innerHTML;
	let html;

	if (!content) return;

	if (docType === 'pdf') {
		return html2pdf()
			.set({
				filename: docName,
			})
			.from(content)
			.toPdf()
			.save();
	}

	if (docType === 'txt') {
		html = container.innerText;
	} else {
		html = `
	<html ${this.doctype === 'doc'
		? 'xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40"'
		: ''
}>
	<head>
		<meta charset="utf-8" />
		<title>${docName}</title>
	</head>
	<body>
		${content}
	</body>
	</html>`;
	}

	const url = `${docType === 'doc'
		? 'data:application/vnd.ms-word;charset=utf-8'
		: 'data:text/plain;charset=utf-8'
	},${encodeURIComponent(html)}`;

	const downloadLink = document.createElement('a');
	document.body.appendChild(downloadLink);
	downloadLink.href = url;
	downloadLink.download = `${docName}.${docType}`;
	downloadLink.click();

	document.body.removeChild(downloadLink);
});

function changeChatTitle(responseId) {
	const $lqdChatUserBubblesLength = document.querySelectorAll('.lqd-chat-user-bubble').length;

	if ($lqdChatUserBubblesLength != 1) return;

	$.ajax({
		type: 'post',
		url: '/dashboard/change-chat-title',
		data: {
			streamed_message_id: responseId,
		},
		success: function (data) {
			if (data.changed) {
				applyStreamedChatTitle(data.chat_id, data.new_title);
			}
		},
	});
}

function applyStreamedChatTitle(chatId, title) {
	const chatTitleEl = document.querySelector(
		`#chat_${chatId} .chat-item-title`,
	);

	if (!chatTitleEl) return;

	const newTitle = title.replaceAll(' ', '\u00a0');
	const newTitleStringArray = newTitle.split('');

	chatTitleEl.innerText = '';

	const interval = setInterval(() => {
		chatTitleEl.innerText += newTitleStringArray.shift();

		if (!newTitleStringArray.length) {
			clearInterval(interval);
		}
	}, 30);
}

function stripTitleFromBubble(responseObj) {
	if (!responseObj.response || !responseObj.response.length) return;

	const combined = responseObj.response.join('');
	const cleaned = combined.replace(/^\s*(<br\s*\/?>|\s)*(```[\w]*\s*(<br\s*\/?>|\s)*)?\{"title"\s*:\s*"[^"]*"[^}]*\}\s*(```\s*)?(<br\s*\/?>|\s)*/i, '');

	if (cleaned !== combined) {
		responseObj.response = [ cleaned ];
	}
}

function stripSuggestionsFromBubble(responseObj) {
	if (!responseObj.response || !responseObj.response.length) return;

	const combined = responseObj.response.join('');
	const cleaned = combined
		.replace(/\s*(<br\s*\/?>)*\s*(```[\w]*\s*(<br\s*\/?>|\s)*)?\{[\s\n]*"suggestions"\s*:\s*\[[\s\S]*?\]\s*\}\s*(```\s*)?(<br\s*\/?>|\s)*$/i, '')
		.trimEnd();

	if (cleaned !== combined) {
		responseObj.response = [ cleaned ];
	}
}

function applySuggestionsToContainer(responseObj) {
	if (!responseObj._streamedSuggestions?.length || !responseObj.bubbleEl) return;

	const container = responseObj.bubbleEl.querySelector('.lqd-chat-bubble-suggestions');
	if (!container) return;

	const alpineData = Alpine?.$data(container);
	if (alpineData) {
		alpineData.suggestions = responseObj._streamedSuggestions;
	}
}

function setChatsCssVars() {
	const chatsWrapper = document.querySelector('.chats-wrap');
	const chatsContainer = document.querySelector('.chats-container');
	const chatsHead = document.querySelector('.lqd-chat-head');
	const chatsForm = document.querySelector('.lqd-chat-form');
	const conversationArea = document.querySelector('.conversation-area');

	if (
		chatsWrapper &&
		chatsContainer &&
		chatsHead &&
		chatsForm &&
		conversationArea
	) {
		chatsWrapper.style.setProperty(
			'--chats-container-height',
			`${conversationArea.offsetHeight - chatsHead.offsetHeight - chatsForm.offsetHeight}px`,
		);
	}
}

(() => {
	setChatsCssVars();

	window.addEventListener('resize', _.debounce(setChatsCssVars, 150));
})();

// Global removeSkillBadge function (used by both skills modal and slash command)
if (typeof window.removeSkillBadge === 'undefined') {
	window.removeSkillBadge = function(id, name) {
		const input = document.getElementById('selected_skill_ids');
		if (input) {
			let ids = input.value.split(',').filter(Boolean).filter(i => i !== id);
			input.value = ids.join(',');
		}

		const chatsV2 = Alpine.store('chatsV2');

		if ( chatsV2 ) {
			chatsV2.removeToolSelection(`skill-${name}`);
		}
	};
}

// Slash Command "/" Skills Shortcut
(() => {
	if (typeof window.__skillsAuth === 'undefined') return;

	let slashDropdownVisible = false;
	let slashSelectedIndex = -1;
	let slashSkills = [];
	let debounceTimer = null;

	function getPromptEl() {
		return document.getElementById('prompt');
	}

	function ensureDropdownExists() {
		let dropdown = document.getElementById('slash-command-dropdown');
		if (!dropdown) {
			dropdown = document.createElement('div');
			dropdown.id = 'slash-command-dropdown';
			dropdown.style.cssText = 'display:none;position:fixed;z-index:1000;';
			dropdown.className = 'lqd-dropdown-dropdown-content border border-transparent dark:border-dropdown-border rounded-dropdown bg-dropdown-background text-dropdown-foreground shadow-lg shadow-black/5';
			dropdown.style.width = '275px';
			dropdown.innerHTML =
				'<div class="border-b relative">' +
					'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="pointer-events-none absolute start-3 top-1/2 size-4 -translate-y-1/2 text-foreground opacity-50"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>' +
					'<input id="slash-command-search" type="text" placeholder="'+ (magicai_localize.search_skills ?? 'Search Skills') +'" class="h-11 w-full border-0 bg-transparent py-0 pe-4 ps-9 text-heading-foreground placeholder:text-foreground/40 focus:outline-none focus:ring-0 sm:text-[12px]" />' +
				'</div>' +
				'<div id="slash-command-list" class="max-h-[200px] overflow-y-auto py-1"></div>' +
				'<button type="button" id="slash-manage-skills-btn" class="flex h-11 w-full items-center justify-start gap-2.5 border-t px-3 text-start text-[12px] text-foreground transition-colors hover:bg-foreground/5">' +
					'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="size-4 opacity-50"><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path d="M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5" /></svg>' +
					(magicai_localize.manage_skills ?? 'Manage Skills') +
				'</button>';
			document.body.appendChild(dropdown);

			dropdown.querySelector('#slash-manage-skills-btn').addEventListener('click', () => {
				hideDropdown();
				window.dispatchEvent(new CustomEvent('open-skills-modal'));
			});

			let slashSearchTimer = null;
			const searchInput = dropdown.querySelector('#slash-command-search');
			searchInput.addEventListener('input', e => {
				const q = e.target.value.trim();
				clearTimeout(slashSearchTimer);
				slashSearchTimer = setTimeout(async () => {
					try {
						const isGuest = !window.__skillsAuth;
						const searchUrl = isGuest ? '/dashboard/user/skills/public-search' : '/dashboard/user/skills/search';
						const res = await fetch(`${searchUrl}?q=${encodeURIComponent(q)}`, {
							headers: { 'X-Requested-With': 'XMLHttpRequest' }
						});
						if (!res.ok) return;
						const data = await res.json();
						const skills = data.skills || [];
						slashSkills = skills;
						slashSelectedIndex = 0;
						const list = getList();
						if (!list) return;
						if (skills.length === 0) {
							list.innerHTML = '<div class="px-4 py-6 text-center"><p class="m-0 text-xs text-foreground/50">No skills found</p></div>';
						} else {
							list.innerHTML = skills.map((skill, i) => {
								const safeName = escapeHtml(skill.name || '');
								const desc = escapeHtml(skill.description || '');
								const truncDesc = desc.length > 35 ? desc.substring(0, 35) + '...' : desc;
								return `<button type="button" class="slash-cmd-item flex w-full flex-col items-start justify-start gap-2 overflow-hidden px-4 py-2 text-start text-[12px] leading-tight text-heading-foreground hover:bg-foreground/5 [&.is-focused]:bg-foreground/5 ${i === 0 ? 'is-focused' : ''}" data-index="${i}" data-id="${skill.id}" data-name="${safeName}" data-desc="${desc}">
									<span class="block w-full font-medium">${safeName}</span>
									<span class="block w-full truncate opacity-70">${truncDesc}</span>
								</button>`;
							}).join('');
						}
					} catch {}
				}, 200);
			});
			searchInput.addEventListener('keydown', e => {
				if (e.key === 'Escape') {
					hideDropdown();
				} else if (e.key === 'Enter') {
					e.preventDefault();
					const visible = [ ...dropdown.querySelectorAll('.slash-cmd-item') ].filter(el => el.style.display !== 'none');
					const highlighted = visible.find(el => el.classList.contains('is-focused')) || visible[0];
					if (highlighted) selectSlashSkill(highlighted.dataset.id, highlighted.dataset.name);
				}
			});
		}
		return dropdown;
	}

	function navigateDropdownItems(direction) {
		const dropdown = document.getElementById('slash-command-dropdown');
		if (!dropdown) return;
		const visible = [ ...dropdown.querySelectorAll('.slash-cmd-item') ].filter(el => el.style.display !== 'none');
		if (!visible.length) return;

		let idx = visible.findIndex(el => el.classList.contains('is-focused'));
		visible.forEach(el => el.classList.remove('is-focused'));

		let next = idx + direction;
		if (next < 0) next = visible.length - 1;
		if (next >= visible.length) next = 0;

		visible[next].classList.add('is-focused');
		visible[next].scrollIntoView({ block: 'nearest' });
		slashSelectedIndex = slashSkills.findIndex(s => String(s.id) === visible[next].dataset.id);
	}

	function getDropdown() {
		return ensureDropdownExists();
	}

	function getList() {
		return document.getElementById('slash-command-list');
	}

	function positionDropdown() {
		const dropdown = getDropdown();
		const promptEl = getPromptEl();
		if (!dropdown || !promptEl) return;

		const inputContainer = promptEl.closest('.lqd-chat-form-inputs-container') || promptEl.parentElement;
		const rect = inputContainer.getBoundingClientRect();

		dropdown.style.left = rect.left + 'px';
		dropdown.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
		dropdown.style.width = Math.min(rect.width, 320) + 'px';
	}

	function showDropdown(skills) {
		const dropdown = getDropdown();
		const list = getList();

		if (!dropdown || !list) return;

		slashSkills = skills;
		slashSelectedIndex = 0;

		if (skills.length === 0) {
			list.innerHTML = '<div class="px-4 py-6 text-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mx-auto mb-1.5 size-6 text-foreground/30"><path d="M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5"/></svg><p class="m-0 text-xs text-foreground/50">No skills added yet</p></div>';
		} else {
			list.innerHTML = skills.map((skill, i) => {
				const safeName = escapeHtml(skill.name || '');
				const desc = escapeHtml(skill.description || '');
				const truncDesc = desc.length > 35 ? desc.substring(0, 35) + '...' : desc;
				return `<button type="button" class="slash-cmd-item flex w-full flex-col items-start justify-start gap-2 overflow-hidden px-4 py-2 text-start text-[12px] leading-tight text-heading-foreground hover:bg-foreground/5 [&.is-focused]:bg-foreground/5 ${i === 0 ? 'is-focused' : ''}" data-index="${i}" data-id="${skill.id}" data-name="${safeName}" data-desc="${desc}">
					<span class="block w-full font-medium">${safeName}</span>
					<span class="block w-full truncate opacity-70">${truncDesc}</span>
				</button>`;
			}).join('');
		}

		const searchInput = dropdown.querySelector('#slash-command-search');

		if (searchInput) searchInput.value = '';

		positionDropdown();
		dropdown.style.display = 'block';
		slashDropdownVisible = true;

		setTimeout(() => searchInput?.focus(), 50);
	}

	function hideDropdown() {
		const dropdown = document.getElementById('slash-command-dropdown');
		if (dropdown) {
			dropdown.style.display = 'none';
			dropdown.blur();
		}
		slashDropdownVisible = false;
		slashSkills = [];
		slashSelectedIndex = -1;
		// getPromptEl()?.focus();
	}

	function escapeHtml(str) {
		const div = document.createElement('div');
		div.textContent = str;
		return div.innerHTML;
	}

	function ensureSkillElements() {
		let input = document.getElementById('selected_skill_ids');
		if (!input) {
			input = document.createElement('input');
			input.id = 'selected_skill_ids';
			input.type = 'hidden';
			input.value = '';
			const chatForm = document.getElementById('chat_form');
			if (chatForm) chatForm.appendChild(input);
		}

		return { input };
	}

	function selectSlashSkill(id, name) {
		// Check if user is authenticated
		if (!window.__skillsAuth) {
			hideDropdown();
			toastr.info('Please login to use skills.');
			return;
		}
		// Check if user's plan includes skills
		if (!window.__skillsPlan) {
			hideDropdown();
			toastr.info('Your current plan does not include Skills. Please upgrade your plan.');
			return;
		}
		// Skills are not compatible with Council Mode
		if (isCouncilModeActive()) {
			hideDropdown();
			toastr.warning('Skills are not supported in Council Mode. Please disable Council Mode to use skills.');
			return;
		}
		const { input } = ensureSkillElements();

		if (input) {
			let ids = input.value ? input.value.split(',').filter(Boolean) : [];
			if (!ids.includes(String(id))) {
				ids.push(String(id));
				input.value = ids.join(',');
			}
		}

		if (window.__registerSkillIdMapping) {
			window.__registerSkillIdMapping(name, id);
		}

		const chatsV2 = Alpine.store('chatsV2');

		if ( chatsV2 ) {
			chatsV2.addToolSelection(`skill-${name}`);
		}

		// Clear the slash from the prompt and sync with Alpine
		const el = getPromptEl();
		if (el) {
			const val = el.value;
			const slashIndex = val.lastIndexOf('/');
			if (slashIndex >= 0) {
				el.value = val.substring(0, slashIndex).trim();
			} else {
				el.value = '';
			}
			el.dispatchEvent(new Event('input', { bubbles: true }));
			el.focus();
		}

		hideDropdown();
	}

	function highlightItem(index) {
		const list = getList();
		if (!list) return;
		list.querySelectorAll('.slash-cmd-item').forEach((item, i) => {
			item.classList.toggle('is-focused', i === index);
		});
	}

	function handleInput(e) {
		const val = e.target.value;

		// Check if user typed "/" at start or after space
		const match = val.match(/(?:^|\s)\/(\S*)$/);
		if (!match) {
			if (slashDropdownVisible) hideDropdown();
			return;
		}

		const query = match[1] || '';

		clearTimeout(debounceTimer);
		debounceTimer = setTimeout(async () => {
			try {
				const isGuest = !window.__skillsAuth;
				const searchUrl = isGuest ? '/dashboard/user/skills/public-search' : '/dashboard/user/skills/search';
				const res = await fetch(`${searchUrl}?q=${encodeURIComponent(query)}`, {
					headers: { 'X-Requested-With': 'XMLHttpRequest' }
				});
				if (!res.ok) { hideDropdown(); return; }
				const data = await res.json();
				showDropdown(data.skills || []);
			} catch {
				hideDropdown();
			}
		}, 200);
	}

	function handleKeydown(e) {
		if (!slashDropdownVisible) return;

		if (e.key === 'ArrowDown') {
			e.preventDefault();
			slashSelectedIndex = Math.min(slashSelectedIndex + 1, slashSkills.length - 1);
			highlightItem(slashSelectedIndex);
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			slashSelectedIndex = Math.max(slashSelectedIndex - 1, 0);
			highlightItem(slashSelectedIndex);
		} else if (e.key === 'Enter' && slashSelectedIndex >= 0) {
			e.preventDefault();
			e.stopPropagation();
			const skill = slashSkills[slashSelectedIndex];
			if (skill) selectSlashSkill(skill.id, skill.name);
		} else if (e.key === 'Escape') {
			e.preventDefault();
			hideDropdown();
		}
	}

	// Use document-level delegation with both input and keyup for maximum compatibility
	document.addEventListener('input', e => {
		if (e.target && e.target.id === 'prompt') {
			handleInput(e);
		}
	}, true);

	document.addEventListener('keyup', e => {
		if (e.target && e.target.id === 'prompt' && !slashDropdownVisible) {
			// Fallback: also check on keyup in case input event didn't fire
			handleInput(e);
		}
	}, true);

	document.addEventListener('keydown', e => {
		if (slashDropdownVisible && (e.target?.id === 'prompt' || e.target?.id === 'slash-command-search')) {
			handleKeydown(e);
		}
	}, true);

	// Also intercept keypress to prevent Enter from submitting chat when dropdown is open
	document.addEventListener('keypress', e => {
		if (slashDropdownVisible && (e.target?.id === 'prompt' || e.target?.id === 'slash-command-search') && e.key === 'Enter') {
			e.preventDefault();
			e.stopImmediatePropagation();
		}
	}, true);

	// Click on dropdown items (use delegation on the dropdown itself)
	document.addEventListener('click', e => {
		// Handle Manage Skills button
		if (e.target.closest('#slash-manage-skills-btn')) {
			e.preventDefault();
			e.stopPropagation();
			hideDropdown();
			window.dispatchEvent(new CustomEvent('open-skills-modal'));
			return;
		}

		// Handle clicks on slash command items via delegation
		const item = e.target.closest('.slash-cmd-item');
		if (item && item.dataset.id) {
			e.preventDefault();
			e.stopPropagation();
			selectSlashSkill(item.dataset.id, item.dataset.name);
			return;
		}

		// Close dropdown when clicking outside
		const dropdown = document.getElementById('slash-command-dropdown');
		if (slashDropdownVisible && dropdown && !dropdown.contains(e.target) && e.target?.id !== 'prompt' && e.target?.id !== 'slash-command-search') {
			hideDropdown();
		}
	}, true);

	// Reposition on window resize
	window.addEventListener('resize', () => {
		if (slashDropdownVisible) positionDropdown();
	});
})();


