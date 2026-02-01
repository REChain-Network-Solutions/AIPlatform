$('#file').on('change', function () {
	'use strict';
	let isInvalid = false;
	const file = this.files[0];
	if (!file) return;

	const allowedExtensions = ['mp3', 'mpeg', 'mpga', 'm4a', 'wav', 'ogg', 'mp4', 'webm', 'aac', 'flac'];
	const allowedMimes = [
		'audio/mpeg', 'audio/mp3', 'audio/x-m4a', 'audio/mp4',
		'audio/wav', 'audio/webm', 'audio/ogg', 'audio/aac', 'audio/flac'
	];

	// MIME type to extension mapping
	const mimeToExt = {
		'audio/mpeg': 'mp3',
		'audio/mp3': 'mp3',
		'audio/x-m4a': 'm4a',
		'audio/mp4': 'm4a',
		'audio/wav': 'wav',
		'audio/webm': 'webm',
		'audio/ogg': 'ogg',
		'audio/aac': 'aac',
		'audio/flac': 'flac'
	};

	if (file.size > 24900000) {
		toastr.error(magicai_localize?.file_size_exceed || 'This file exceeds the upload limit');
		isInvalid = true;
	}

	let mime = (file.type || '').toLowerCase();
	let name = file.name || '';
	let ext = '';

	// Extract extension from filename
	if (name.includes('.')) {
		ext = name.split('.').pop().toLowerCase();
	}

	// If no extension found, derive from MIME type
	if (!ext && mime && mimeToExt[mime]) {
		ext = mimeToExt[mime];
		console.log(`No extension in filename, derived '${ext}' from MIME type '${mime}'`);
	}

	console.log('MIME:', mime);
	console.log('Name:', name);
	console.log('Extension:', ext);

	// Validate: must pass EITHER mime OR extension check
	if (!allowedMimes.includes(mime) && !allowedExtensions.includes(ext)) {
		toastr.error(
			magicai_localize?.invalid_extension ||
			'Invalid audio file. Accepted: mp3, mpeg, mpga, m4a, wav, ogg, webm, aac, flac'
		);
		isInvalid = true;
	}

	// Block video/webm specifically
	if (!isInvalid && mime === 'video/webm') {
		toastr.error(
			magicai_localize?.invalid_extension ||
			'Video files are not allowed. Please upload an audio-only file.'
		);
		isInvalid = true;
	}

	if (isInvalid) {
		this.value = null;
	}
});

// @formatter:off
document.addEventListener( 'DOMContentLoaded', function () {
	'use strict';

	var el = document.getElementById( 'language' );

	if (el) {
		window.TomSelect && ( new TomSelect( el, {
			copyClassesToDropdown: false,
			dropdownClass: 'dropdown-menu ts-dropdown',
			optionClass: 'dropdown-item',
			controlInput: '<input>',
			render: {
				item: function ( data, escape ) {
					if ( data.customProperties ) {
						return '<div><span class="dropdown-item-indicator">' + data.customProperties + '</span>' + escape( data.text ) + '</div>';
					}
					return '<div>' + escape( data.text ) + '</div>';
				},
				option: function ( data, escape ) {
					if ( data.customProperties ) {
						return '<div><span class="dropdown-item-indicator">' + data.customProperties + '</span>' + escape( data.text ) + '</div>';
					}
					return '<div>' + escape( data.text ) + '</div>';
				},
			},
		} ) );
	}

} );
// @formatter:on

function fillAnExample(selector){
	'use strict';

	const prompts = [
		'Cityscape at sunset in retro vector illustration',
		'Painting of a flower vase on a kitchen table with a window in the backdrop.',
		'Memphis style painting of a flower vase on a kitchen table with a window in the backdrop.',
		'Illustration of a cat sitting on a couch in a living room with a coffee mug in its hand.',
		'Delicious pizza with all the toppings.',
		'a super detailed infographic of a working time machine 8k',
		'hedgehog smelling a flower',
		'Freeform ferrofluids, beautiful dark chaos',
		'a home built in a huge Soap bubble, windows',
		'photo of an extremely cute alien fish swimming an alien habitable underwater planet'
	];

	var item = prompts[Math.floor(Math.random()*prompts.length)];

	$('.' + selector).val(item);

	return false;
}
