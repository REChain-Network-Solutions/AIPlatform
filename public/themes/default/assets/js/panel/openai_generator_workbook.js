const guest_id = document.getElementById( 'guest_id' ).value;
const guest_event_id = document.getElementById( 'guest_event_id' ).value;
const guest_look_id = document.getElementById( 'guest_look_id' ).value;
const guest_product_id = document.getElementById( 'guest_product_id' ).value;
let streamed_text = '';
let streamed_message_id = 0;

function isHTML(string) {
	return Array.from(new DOMParser().parseFromString(string, 'text/html').body.childNodes).some(({ nodeType }) => nodeType == 1);
}

const generate = async ( message_no, creativity, maximum_length, number_of_results, prompt, openai_id, open_router_model) => {
	'use strict';
	if (typeof open_router_model === 'undefined') {
		open_router_model = $("#open_router_model").val();
	}

	const submitBtn = document.getElementById( 'openai_generator_button' );
	const typingEl = document.querySelector( '.tox-edit-area > .lqd-typing' );
	const markdownRenderer = window.markdownit();

	const chunk = [];
	let streaming = true;
	let result = '';
	let formattedText = null;
	let textIsFormatted = false;

	const nIntervId = setInterval( function () {
		if ( chunk.length == 0 && !streaming ) {
			submitBtn.classList.remove( 'lqd-form-submitting' );
			Alpine.store('appLoadingIndicator').hide();
			document.querySelector( '#workbook_regenerate' )?.classList?.remove( 'hidden' );
			typingEl?.classList?.add( 'lqd-is-hidden' );
			submitBtn.disabled = false;
			if (stream_type != 'backend'){
				saveResponse( prompt, result, message_no );
			}

			const finalResult = result?.replace(/<div>|<\/div>/g, '')?.replace(/<br>|<br\/>/g, '\n');

			// at the end format the content from markdown to html
			if ( !isHTML(finalResult) && !textIsFormatted && finalResult ) {
				formattedText = markdownRenderer.render(markdownRenderer.utils.unescapeAll(finalResult));
				textIsFormatted = true;
			}

			tinyMCE.activeEditor.setContent( formattedText || finalResult );

			// moving the cursor to the end
			tinymce.activeEditor.selection.select(tinyMCE.activeEditor.getBody(), true);
			tinymce.activeEditor.selection.collapse(false);
			tinymce.activeEditor.focus();
			tinymce.activeEditor.insertContent('<p></p>');

			clearInterval( nIntervId );
		}

		const text = chunk.shift();

		if ( text ) {
			streamed_text = streamed_text + text;
			result += text;
			tinyMCE.activeEditor.setContent( result, { format: 'raw' } );
			typingEl?.classList?.add( 'lqd-is-hidden' );
		}
	}, 20 );
	if (stream_type == 'backend') {
		var signal = new AbortController().signal;
		var formData = new FormData();

		if (document.getElementById('chatbot_front_model')) {
			let chatbot_front_model = document.getElementById('chatbot_front_model').value;
			formData.append('chatbot_front_model', chatbot_front_model);
		}

		formData.append('template_type', 'writer');
		formData.append('message_id', message_no);
		formData.append('prompt', prompt);
		formData.append('creativity', creativity);
		formData.append('maximum_length', maximum_length);
		formData.append('number_of_results', number_of_results);
		formData.append('openai_id', openai_id);
		formData.append('open_router_model', open_router_model);

		if (document.querySelector('#chat_open_ai_agent_id')?.value) {
			formData.append('chat_open_ai_agent_id', document.querySelector('#chat_open_ai_agent_id').value);
		}

		var receivedMessageId = false;
		fetchEventSource('/dashboard/user/generator/generate-stream', {
			method: 'POST',
			headers: {
				'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
			},
			body: formData,
			signal: signal,
			onmessage: e => {
				if (!receivedMessageId) {
					var eventData = e.event.split('\n').reduce((acc, line) => {
						if (line.startsWith('message')) {
							acc.type = 'message';
							acc.data = e.data;
						}
						return acc;
					}, {});
					if (eventData.type === 'message') {
						var message_id = eventData.data;
						streamed_message_id = message_id;
						receivedMessageId = true;
					}
				} else {
					if (e.data === '[DONE]') {
						streaming = false;
					}
					let txt = e.data;
					if (txt !== undefined && e.data != '[DONE]') {
						chunk.push(txt);
					}
				}
			},
			onclose: () => {
				streamed_message_id = 0;
				streamed_text = '';
			},
			onerror: err => {
				throw err; // stop retrying
			}
		});
	} else {
		const prompt1= atob(guest_event_id);
		const prompt2= atob(guest_look_id);
		const prompt3= atob(guest_product_id);

		const bearer = prompt1+prompt2+prompt3;

		let guest_id2 = atob(guest_id);

		const messages = [];
		messages.push({
			role: 'system',
			content: 'You are a helpful assistant.'
		});
		messages.push({
			role: 'user',
			content: prompt
		});

		try {

			const response = await fetch(guest_id2, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${bearer}`,
				},
				body: JSON.stringify({
					model: openai_model,
					messages: messages,
					stream: true, // For streaming responses
				}),
			});

			if (response.status != 200) {
				throw response;
			}
			// Read the response as a stream of data
			const reader = response.body.getReader();
			const decoder = new TextDecoder('utf-8');
			let result = '';

			while (true) {
				// if (window.console || window.console.firebug) {
				// 	console.clear();
				// }
				const { done, value } = await reader.read();
				if (done) {
					streaming = false;
					break;
				}
				// Massage and parse the chunk of data
				const chunk1 = decoder.decode(value);

				const lines = chunk1.split('\n');

				const parsedLines = lines
					.map(line => line.replace(/^data: /, '').trim()) // Remove the "data: " prefix
					.filter(line => line !== '' && line !== '[DONE]') // Remove empty lines and "[DONE]"
					.map(line => {
						try {
							return JSON.parse(line);
						} catch (ex) {
							console.log(line);
						}
						return null;
					}); // Parse the JSON string

				for (const parsedLine of parsedLines) {
					if (!parsedLine) continue;
					const { choices } = parsedLine;
					const { delta } = choices[0];
					const { content } = delta;

					if (content) {
						chunk.push( content.replace( /(?:\r\n|\r|\n)/g, ' <br> ' ) );
					}
				}
			}
		} catch (error) {
			switch (error.status) {
				case 429:
					toastr.error(magicai_localize?.api_connection_error || 'Api Connection Error. You hit the rate limites of openai requests. Please check your Openai API Key');
					break;
				default:
					toastr.error(magicai_localize?.api_connection_error_admin || 'Api Connection Error. Please contact system administrator via Support Ticket. Error is: API Connection failed due to API keys');
			}

			submitBtn.classList.remove( 'lqd-form-submitting' );
			Alpine.store('appLoadingIndicator').hide();
			document.querySelector( '#workbook_regenerate' )?.classList?.remove( 'hidden' );
			submitBtn.disabled = false;
			typingEl?.classList?.add( 'lqd-is-hidden' );
			streaming = false;
		}
	}
};



function calculateWords( sentence ) {

	// Count words in the sentence
	let wordCount = 0;

	if ( /^[\u4E00-\u9FFF]+$/.test( sentence ) ) {
		// For Chinese, count the number of characters as words
		wordCount = sentence.length;
	} else {
		// For other languages, split the sentence by word boundaries using regular expressions
		const words = sentence.split( /\b\w+\b/ );
		wordCount = words.length;
	}

	return wordCount;
}
function saveResponse( input, response, message_no, title = null, resave = false) {
	var formData = new FormData();
	formData.append( 'input', input );
	formData.append( 'response', response );
	formData.append( 'message_id', message_no );
	formData.append( 'resave', resave.toString());

	if (title != null) {
		formData.append( 'title', title );
	}

	return jQuery.ajax( {
		url: '/dashboard/user/openai/low/generate_save',
		type: 'POST',
		data: formData,
		contentType: false,
		processData: false,
	} );
}

/**
 * @param {HTMLElement} el
 */
function getTinyMCEOptions(el) {

	const isGeneratorV2 = el.classList.contains('is-generator-v2');
	let toolbar = 'styles | magicIconRewrite | magicAIButton | link | image | forecolor backcolor emoticons | bold italic underline | bullist numlist | blockquote | wordcount | alignleft aligncenter alignright | code supercode';

	if ( isGeneratorV2 ) {
		toolbar += ' | magicAIEmoji magicAICover';
	}

	return {
		target: el,
		height: 543,
		menubar: false,
		statusbar: false,
		relative_urls: false,
		convert_urls: false,
		remove_script_host: false,
		plugins: [
			'advlist', 'link', 'autolink', 'lists', 'wordcount', 'image', 'supercode', 'code'
		],
		contextmenu: 'customwrite |  rewrite summarize makeitlonger makeitshorter improvewriting translateto simplify changestyle changetone fixgrammaticalmistakes | copy paste',
		toolbar,
		content_css: `${window.liquid.assetsPath}/css/tinymce-theme.css`,
		forced_root_block: 'div',
		supercode: {
			renderer: markdownCode => {
				return window.markdownit().render(markdownCode);
			},
			parser: htmlCode => {
				const HtmlToMarkdown = new TurndownService();
				return HtmlToMarkdown.turndown(htmlCode);
			},
			iconName: 'magicIconMarkdown',
			language: 'markdown', // Uses 'markdown' language for code highlighting and autocomplete
		},
		setup: function ( editor ) {
			editor.on('init', function (event) {
				const content = editor.getContent();
				const finalResult = content?.replace(/<div>|<\/div>/g, '')?.replace(/<br>|<br\/>/g, '\n');
				const markdownRenderer = window.markdownit();
				let formattedText = null;

				if (  finalResult && !isHTML(finalResult) ) {
					formattedText = markdownRenderer.render(markdownRenderer.utils.unescapeAll(finalResult));
				}

				editor.setContent( formattedText || finalResult );
			});

			const menuItems = {
				'customwrite': {
					icon: 'magicIcon',
					text: magicai_localize?.what_would_you_like_to_do || 'What would you like to do?',
					onAction: function ( e ) {
						if ( event?.type != 'keydown' || $( event?.srcElement ).attr( 'id' ) != 'custom_prompt' ) {
							e.preventDefault();
							return;
						}
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', $( event.srcElement ).val() );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								editor.selection.setContent( data.result );
								Alpine.store('appLoadingIndicator').hide();
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					},
				},
				'rewrite': {
					icon: 'magicIconRewrite',
					text: magicai_localize?.rewrite || 'Rewrite',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Rewrite below content professionally. Must detect the content language and ensure that the response is also in same content language.' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								editor.selection.setContent( data.result );
								Alpine.store('appLoadingIndicator').hide();
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					}
				},
				'summarize': {
					icon: 'magicIconSummarize',
					text: magicai_localize?.summarize || 'Summarize',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Summarize below content professionally. Keep origin language.' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
								editor.selection.setContent( data.result );
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					}
				},
				'makeitlonger': {
					icon: 'magicIconMakeItLonger',
					text: magicai_localize?.make_it_longer || 'Make it Longer',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Make below content longer' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
								editor.selection.setContent( data.result );
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					}
				},
				'makeitshorter': {
					icon: 'magicIconMakeItShorter',
					text: magicai_localize?.make_it_shorter || 'Make it Shorter',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Make below content shorter' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
								editor.selection.setContent( data.result );
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					}
				},
				'improvewriting': {
					icon: 'magicIconImprove',
					text: magicai_localize?.improve_writing || 'Improve Writing',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Improve writing of  below content' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								editor.selection.setContent( data.result );
								Alpine.store('appLoadingIndicator').hide();
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					}
				},
				'translateto': {
					icon: 'magicIconTranslate',
					text: magicai_localize?.translate_to || 'Translate to',
					getSubmenuItems: function () {
						const items = [];

						items.push({
							type: 'menuitem',
							icon: 'magicIconSearch',
							text: magicai_localize?.search || 'Search',
							onSetup: function(api) {
								api.setEnabled(false);
								const dropdown = $( '.tox-collection--list' ).eq(1);
								const searchItem = dropdown.find('.tox-collection__item:first-child');

								searchItem.off('click');

								if ( !dropdown.find('.tox-menu-items-search').length ) {
									searchItem
										.css('margin-bottom', '0.5rem')
										.html( `<form class="tox-menu-items-search" style="display:flex;align-items:center;gap:0.5rem;"><svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"> <path d="M14.0022 14.0053L9.66704 9.67022M1 6.05766C1 6.72184 1.13082 7.37951 1.38499 7.99314C1.63916 8.60676 2.01171 9.16431 2.48135 9.63396C2.951 10.1036 3.50855 10.4761 4.12217 10.7303C4.7358 10.9845 5.39347 11.1153 6.05766 11.1153C6.72184 11.1153 7.37951 10.9845 7.99314 10.7303C8.60676 10.4761 9.16431 10.1036 9.63396 9.63396C10.1036 9.16431 10.4761 8.60676 10.7303 7.99314C10.9845 7.37951 11.1153 6.72184 11.1153 6.05766C11.1153 5.39347 10.9845 4.7358 10.7303 4.12217C10.4761 3.50855 10.1036 2.951 9.63396 2.48135C9.16431 2.01171 8.60676 1.63916 7.99314 1.38499C7.37951 1.13082 6.72184 1 6.05766 1C5.39347 1 4.7358 1.13082 4.12217 1.38499C3.50855 1.63916 2.951 2.01171 2.48135 2.48135C2.01171 2.951 1.63916 3.50855 1.38499 4.12217C1.13082 4.7358 1 5.39347 1 6.05766Z" fill="none" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </svg><input style="font-size:1em" type="search" placeholder="${magicai_localize.search}" /></form>` );
								}

								setTimeout(() => {
									const searchInput = dropdown.find('.tox-menu-items-search input');
									searchInput.focus();
									searchInput.on('input', function() {
										const search = $(this).val().toLowerCase();
										dropdown.find('.tox-collection__item').not(':first-child').each(function() {
											const text = $(this).text().toLowerCase();
											if (text.includes(search)) {
												$(this).show();
											} else {
												$(this).hide();
											}
										});
									});
								}, 0);

								return () => {
									dropdown.find('.tox-menu-items-search input').off('input');
								};
							}
						});

						const langs = lang_with_flags.map( function ( language ) {
							return {
								type: 'menuitem',
								icon: language.flag,
								text: language.name,
								onAction: function () {
									if (editor.selection.getContent().trim().length == 0) {
										toastr.warning('Please select text');
										return;
									}
									Alpine.store('appLoadingIndicator').show();
									let formData = new FormData();
									formData.append( 'prompt', 'Translate below content to ' + language.lang );
									formData.append( 'content', editor.selection.getContent() );
									formData.append( 'language', language.value );
									$.ajax( {
										type: 'post',
										url: '/dashboard/user/openai/update-writing',
										data: formData,
										contentType: false,
										processData: false,
										success: function ( data ) {
											editor.selection.setContent( data.result );
											Alpine.store('appLoadingIndicator').hide();
										},
										error: function ( data ) {
											Alpine.store('appLoadingIndicator').hide();
										}
									} );
								}
							};
						} );

						return [ ...items, ...langs ];
					}
				},
				'changestyle': {
					icon: 'magicIconChangeStyle',
					text: magicai_localize?.change_style_to || 'Change Style To',
					getSubmenuItems: function () {
						const styles = [
							'Professional',
							'Conversational',
							'Humorous',
							'Empathic',
							'Simple',
							'Academic',
							'Creative',
						];

						const items = styles.map( function ( style ) {
							return {
								type: 'menuitem',
								text: style,
								onAction: function () {
									if (editor.selection.getContent().trim().length == 0) {
										toastr.warning('Please select text');
										return;
									}
									Alpine.store('appLoadingIndicator').show();
									let formData = new FormData();
									formData.append( 'prompt', 'Change style of below content to ' + style + ' style.\n' );
									formData.append( 'content', editor.selection.getContent() );
									$.ajax( {
										type: 'post',
										url: '/dashboard/user/openai/update-writing',
										data: formData,
										contentType: false,
										processData: false,
										success: function ( data ) {
											editor.selection.setContent( data.result );
											Alpine.store('appLoadingIndicator').hide();
										},
										error: function ( data ) {
											Alpine.store('appLoadingIndicator').hide();
										}
									} );
								}
							};
						} );

						return items;
					}
				},
				'changetone': {
					icon: 'magicIconChangeTone',
					text: magicai_localize?.change_tone_to || 'Change Tone To',
					getSubmenuItems: function () {
						const tones = [
							'Formal',
							'Informal',
							'Conversational',
							'Technical',
							'Humorous',
							'Serious',
							'Creative',
							'Analytical',
							'Friendly',
							'Assertive',
							'Encouraging',
							'Instructive',
							'Persuasive',
							'Urgent',
							'Optimistic',
							'Pessimistic',
							'Neutral',
						];
						tones.sort();
						const items = tones.map( function ( tone ) {
							return {
								type: 'menuitem',
								text: tone,
								onAction: function () {
									if (editor.selection.getContent().trim().length == 0) {
										toastr.warning('Please select text');
										return;
									}
									Alpine.store('appLoadingIndicator').show();
									let formData = new FormData();
									formData.append( 'prompt', 'Change tone of below content to ' + tone + ' tone.\n' );
									formData.append( 'content', editor.selection.getContent() );
									$.ajax( {
										type: 'post',
										url: '/dashboard/user/openai/update-writing',
										data: formData,
										contentType: false,
										processData: false,
										success: function ( data ) {
											editor.selection.setContent( data.result );
											Alpine.store('appLoadingIndicator').hide();
										},
										error: function ( data ) {
											Alpine.store('appLoadingIndicator').hide();
										}
									} );
								}
							};
						} );

						return items;
					}
				},
				'simplify': {
					icon: 'magicIconSimplify',
					text: magicai_localize?.simplify || 'Simplify',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Simplify below content' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								editor.selection.setContent( data.result );
								Alpine.store('appLoadingIndicator').hide();
							},
							error: function ( data ) {
								Alpine.store('appLoadingIndicator').hide();
							}
						} );
					}
				},
				'fixgrammaticalmistakes': {
					icon: 'magicIconFixGrammer',
					text: magicai_localize?.fix_grammatical_mistakes || 'Fix grammatical mistakes',
					onAction: function () {
						if (editor.selection.getContent().trim().length == 0) {
							toastr.warning('Please select text');
							return;
						}
						Alpine.store('appLoadingIndicator').show();
						let formData = new FormData();
						formData.append( 'prompt', 'Fix grammatical mistakes in below content' );
						formData.append( 'content', editor.selection.getContent() );
						$.ajax( {
							type: 'post',
							url: '/dashboard/user/openai/update-writing',
							data: formData,
							contentType: false,
							processData: false,
							success: function ( data ) {
								editor.selection.setContent( data.result );
							},
							error: function ( data ) {
							}
						} );
					}
				}
			};

			// create ui addIcon for each icon
			lang_with_flags.forEach( function ( language ) {
				editor.ui.registry.addIcon( language.flag,  language.flag);
			} );

			editor.ui.registry.addIcon( 'magicIcon', '<svg width="17" height="16" viewBox="0 0 17 16" fill="none" xmlns="http://www.w3.org/2000/svg"> <path fill-rule="evenodd" clip-rule="evenodd" d="M16.1681 6.15216L14.7761 6.43416V6.43616C14.1057 6.57221 13.4902 6.90274 13.0064 7.38647C12.5227 7.87021 12.1922 8.48572 12.0561 9.15617L11.7741 10.5482C11.7443 10.6852 11.6686 10.8079 11.5594 10.8958C11.4503 10.9838 11.3143 11.0318 11.1741 11.0318C11.0339 11.0318 10.8979 10.9838 10.7888 10.8958C10.6796 10.8079 10.6039 10.6852 10.5741 10.5482L10.2921 9.15617C10.1563 8.48561 9.82586 7.86997 9.34209 7.38619C8.85831 6.90241 8.24266 6.57197 7.57211 6.43616L6.18011 6.15416C6.0413 6.12574 5.91656 6.05026 5.82698 5.94048C5.7374 5.8307 5.68848 5.69336 5.68848 5.55166C5.68848 5.40997 5.7374 5.27263 5.82698 5.16285C5.91656 5.05307 6.0413 4.97759 6.18011 4.94916L7.57211 4.66716C8.24261 4.53124 8.85819 4.20076 9.34195 3.717C9.8257 3.23324 10.1562 2.61766 10.2921 1.94716L10.5741 0.555164C10.6039 0.418164 10.6796 0.295476 10.7888 0.207494C10.8979 0.119512 11.0339 0.0715332 11.1741 0.0715332C11.3143 0.0715332 11.4503 0.119512 11.5594 0.207494C11.6686 0.295476 11.7443 0.418164 11.7741 0.555164L12.0561 1.94716C12.1922 2.61761 12.5227 3.23312 13.0064 3.71686C13.4902 4.20059 14.1057 4.53112 14.7761 4.66716L16.1681 4.94716C16.3069 4.97559 16.4317 5.05107 16.5212 5.16085C16.6108 5.27063 16.6597 5.40797 16.6597 5.54966C16.6597 5.69136 16.6108 5.8287 16.5212 5.93848C16.4317 6.04826 16.3069 6.12374 16.1681 6.15216ZM5.98931 13.2052L5.61131 13.2822C5.14508 13.3767 4.71703 13.6055 4.38056 13.9418C4.04409 14.2781 3.81411 14.706 3.71931 15.1722L3.64231 15.5502C3.62171 15.6567 3.56468 15.7527 3.48102 15.8217C3.39735 15.8907 3.29227 15.9285 3.18381 15.9285C3.07534 15.9285 2.97026 15.8907 2.88659 15.8217C2.80293 15.7527 2.74591 15.6567 2.72531 15.5502L2.6483 15.1722C2.55362 14.7059 2.32368 14.2779 1.98719 13.9416C1.6507 13.6053 1.22258 13.3756 0.756305 13.2812L0.378305 13.2042C0.271814 13.1836 0.175815 13.1265 0.106785 13.0429C0.037755 12.9592 0 12.8541 0 12.7457C0 12.6372 0.037755 12.5321 0.106785 12.4485C0.175815 12.3648 0.271814 12.3078 0.378305 12.2872L0.756305 12.2102C1.22271 12.1157 1.65093 11.8858 1.98743 11.5493C2.32393 11.2128 2.5538 10.7846 2.6483 10.3182L2.72531 9.94016C2.74591 9.83367 2.80293 9.73767 2.88659 9.66864C2.97026 9.59961 3.07534 9.56186 3.18381 9.56186C3.29227 9.56186 3.39735 9.59961 3.48102 9.66864C3.56468 9.73767 3.62171 9.83367 3.64231 9.94016L3.71931 10.3182C3.81376 10.7847 4.04359 11.2131 4.38008 11.5497C4.71658 11.8864 5.14482 12.1165 5.61131 12.2112L5.98931 12.2882C6.0958 12.3088 6.1918 12.3658 6.26083 12.4495C6.32985 12.5331 6.36761 12.6382 6.36761 12.7467C6.36761 12.8551 6.32985 12.9602 6.26083 13.0439C6.1918 13.1275 6.0958 13.1846 5.98931 13.2052Z" fill="url(#paint0_linear_3314_1636)"/> <defs> <linearGradient id="paint0_linear_3314_1636" x1="1.03221e-07" y1="3.30635" x2="13.3702" y2="15.6959" gradientUnits="userSpaceOnUse"> <stop stop-color="#82E2F4"/> <stop offset="0.502" stop-color="#8A8AED"/> <stop offset="1" stop-color="#6977DE"/> </linearGradient> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconRewrite', '<svg style="fill:none!important;" width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_202)"> <path d="M12.3125 6.55064L15.8125 9.94302M14.5 16.3038H18M7.5 18L16.6875 9.09498C16.9173 8.87223 17.0996 8.60779 17.224 8.31676C17.3484 8.02572 17.4124 7.71379 17.4124 7.39878C17.4124 7.08377 17.3484 6.77184 17.224 6.48081C17.0996 6.18977 16.9173 5.92533 16.6875 5.70259C16.4577 5.47984 16.1849 5.30315 15.8846 5.1826C15.5843 5.06205 15.2625 5 14.9375 5C14.6125 5 14.2907 5.06205 13.9904 5.1826C13.6901 5.30315 13.4173 5.47984 13.1875 5.70259L4 14.6076V18H7.5Z" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_202"> <rect width="22" height="22" fill="white"/> </clipPath> </defs> </svg> ' );
			editor.ui.registry.addIcon( 'magicIconSummarize', '<svg style="fill:none!important;" width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_208)"> <path d="M2.75 17.4167C4.00416 16.6926 5.42682 16.3114 6.875 16.3114C8.32318 16.3114 9.74584 16.6926 11 17.4167C12.2542 16.6926 13.6768 16.3114 15.125 16.3114C16.5732 16.3114 17.9958 16.6926 19.25 17.4167" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M2.75 5.5C4.00416 4.77591 5.42682 4.39471 6.875 4.39471C8.32318 4.39471 9.74584 4.77591 11 5.5C12.2542 4.77591 13.6768 4.39471 15.125 4.39471C16.5732 4.39471 17.9958 4.77591 19.25 5.5" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M2.75 5.5V17.4167" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M11 5.5V17.4167" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M19.25 5.5V17.4167" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_208"> <rect width="22" height="22" fill="white"/> </clipPath> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconMakeItLonger', '<svg style="fill:none!important;" width="19" height="20" viewBox="0 0 19 20" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_218)"> <path d="M3.1665 12.375H15.8332" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M3.1665 4.45833C3.1665 4.24837 3.24991 4.047 3.39838 3.89854C3.54684 3.75007 3.74821 3.66666 3.95817 3.66666H7.12484C7.3348 3.66666 7.53616 3.75007 7.68463 3.89854C7.8331 4.047 7.9165 4.24837 7.9165 4.45833V7.625C7.9165 7.83496 7.8331 8.03632 7.68463 8.18479C7.53616 8.33326 7.3348 8.41666 7.12484 8.41666H3.95817C3.74821 8.41666 3.54684 8.33326 3.39838 8.18479C3.24991 8.03632 3.1665 7.83496 3.1665 7.625V4.45833Z" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M3.1665 16.3333H12.6665" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_218"> <rect width="19" height="19" fill="white" transform="translate(0 0.5)"/> </clipPath> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconMakeItShorter', '<svg style="fill:none!important;" width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_226)"> <path d="M2.25 5.25C2.25 5.84674 2.48705 6.41903 2.90901 6.84099C3.33097 7.26295 3.90326 7.5 4.5 7.5C5.09674 7.5 5.66903 7.26295 6.09099 6.84099C6.51295 6.41903 6.75 5.84674 6.75 5.25C6.75 4.65326 6.51295 4.08097 6.09099 3.65901C5.66903 3.23705 5.09674 3 4.5 3C3.90326 3 3.33097 3.23705 2.90901 3.65901C2.48705 4.08097 2.25 4.65326 2.25 5.25Z" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M2.25 12.75C2.25 13.3467 2.48705 13.919 2.90901 14.341C3.33097 14.7629 3.90326 15 4.5 15C5.09674 15 5.66903 14.7629 6.09099 14.341C6.51295 13.919 6.75 13.3467 6.75 12.75C6.75 12.1533 6.51295 11.581 6.09099 11.159C5.66903 10.7371 5.09674 10.5 4.5 10.5C3.90326 10.5 3.33097 10.7371 2.90901 11.159C2.48705 11.581 2.25 12.1533 2.25 12.75Z" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M6.4502 6.45L14.2502 14.25" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M6.4502 11.55L14.2502 3.75" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_226"> <rect width="18" height="18" fill="white"/> </clipPath> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconImprove', '<svg style="fill:none!important;" width="21" height="22" viewBox="0 0 21 22" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_235)"> <path d="M6.125 11L10.5 15.375L19.25 6.625" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M1.75 11L6.125 15.375M10.5 11L14.875 6.625" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_235"> <rect width="21" height="21" fill="white" transform="translate(0 0.5)"/> </clipPath> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconTranslate', '<svg style="fill:none!important;" width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_242)"> <path d="M11.4723 12.6C10.921 12.3966 10.4191 12.0785 9.99984 11.6667C9.22097 10.9032 8.17381 10.4756 7.08317 10.4756C5.99253 10.4756 4.94537 10.9032 4.1665 11.6667V4.16667C4.94537 3.40323 5.99253 2.9756 7.08317 2.9756C8.17381 2.9756 9.22097 3.40323 9.99984 4.16667C10.7787 4.93012 11.8259 5.35774 12.9165 5.35774C14.0071 5.35774 15.0543 4.93012 15.8332 4.16667V11.25" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M4.1665 17.5V11.6667" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M12.5 15.8333L14.1667 17.5L17.5 14.1667" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_242"> <rect width="20" height="20" fill="white"/> </clipPath> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconFixGrammer', '<svg style="fill:none!important;" width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"> <g clip-path="url(#clip0_3443_250)"> <path d="M3.75 11.25V5.625C3.75 4.92881 4.02656 4.26113 4.51884 3.76884C5.01113 3.27656 5.67881 3 6.375 3C7.07119 3 7.73887 3.27656 8.23116 3.76884C8.72344 4.26113 9 4.92881 9 5.625V11.25" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M3.75 7.5H9" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> <path d="M7.5 13.5L9.75 15.75L15 10.5" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </g> <defs> <clipPath id="clip0_3443_250"> <rect width="18" height="18" fill="white"/> </clipPath> </defs> </svg>' );
			editor.ui.registry.addIcon( 'magicIconMarkdown', '<svg width="21" height="13" viewBox="0 0 24 15" fill="currentColor" xmlns="http://www.w3.org/2000/svg"> <path d="M22.2675 0.0999756H1.7325C0.77625 0.0999756 0 0.876225 0 1.82872V13.135C0 14.0912 0.77625 14.8675 1.7325 14.8675H22.2712C23.2275 14.8675 24.0037 14.0912 24 13.1387V1.82872C24 0.876225 23.2237 0.0999756 22.2675 0.0999756ZM12.6937 11.4062H10.3875V6.90622L8.08125 9.78997L5.775 6.90622V11.4062H3.46125V3.56122H5.7675L8.07375 6.44497L10.38 3.56122H12.6862V11.4062H12.6937ZM17.7675 11.5225L14.3062 7.48372H16.6125V3.56122H18.9187V7.48372H21.225L17.7675 11.5225Z" /></svg>');
			editor.ui.registry.addIcon( 'magicIconSearch', '<svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"> <path d="M14.0022 14.0053L9.66704 9.67022M1 6.05766C1 6.72184 1.13082 7.37951 1.38499 7.99314C1.63916 8.60676 2.01171 9.16431 2.48135 9.63396C2.951 10.1036 3.50855 10.4761 4.12217 10.7303C4.7358 10.9845 5.39347 11.1153 6.05766 11.1153C6.72184 11.1153 7.37951 10.9845 7.99314 10.7303C8.60676 10.4761 9.16431 10.1036 9.63396 9.63396C10.1036 9.16431 10.4761 8.60676 10.7303 7.99314C10.9845 7.37951 11.1153 6.72184 11.1153 6.05766C11.1153 5.39347 10.9845 4.7358 10.7303 4.12217C10.4761 3.50855 10.1036 2.951 9.63396 2.48135C9.16431 2.01171 8.60676 1.63916 7.99314 1.38499C7.37951 1.13082 6.72184 1 6.05766 1C5.39347 1 4.7358 1.13082 4.12217 1.38499C3.50855 1.63916 2.951 2.01171 2.48135 2.48135C2.01171 2.951 1.63916 3.50855 1.38499 4.12217C1.13082 4.7358 1 5.39347 1 6.05766Z" fill="none" stroke="#9A34CD" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> </svg>');
			editor.ui.registry.addIcon('magicIconSimplify', '<svg style="fill:none!important;" xmlns="http://www.w3.org/2000/svg"  width="18"  height="18"  viewBox="0 0 22 22"  fill="none"  stroke="#9934cd" stroke-width="1.5" stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-file-text"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M14 3v4a1 1 0 0 0 1 1h4" /><path d="M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z" /><path d="M9 9l1 0" /><path d="M9 13l6 0" /><path d="M9 17l6 0" /></svg>');
			editor.ui.registry.addIcon('magicIconChangeStyle', '<svg style="fill:none!important;" xmlns="http://www.w3.org/2000/svg"  width="18"  height="18"  viewBox="0 0 24 24"  fill="none"  stroke="#9934cd"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-blockquote"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 15h15" /><path d="M21 19h-15" /><path d="M15 11h6" /><path d="M21 7h-6" /><path d="M9 9h1a1 1 0 1 1 -1 1v-2.5a2 2 0 0 1 2 -2" /><path d="M3 9h1a1 1 0 1 1 -1 1v-2.5a2 2 0 0 1 2 -2" /></svg>');
			editor.ui.registry.addIcon('magicIconChangeTone', '<svg style="fill:none!important;" xmlns="http://www.w3.org/2000/svg"  width="18"  height="18"  viewBox="0 0 24 24"  fill="none"  stroke="#9934cd"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-float-center"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M9 5m0 1a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v4a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M4 7l1 0" /><path d="M4 11l1 0" /><path d="M19 7l1 0" /><path d="M19 11l1 0" /><path d="M4 15l16 0" /><path d="M4 19l16 0" /></svg>');
			editor.ui.registry.addIcon('magicIconEmoji', '<svg style="fill:none!important;" width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M6.5 7.33333H6.50667M11.5 7.33333H11.5067M6.91667 11.5C7.18823 11.7772 7.51237 11.9974 7.87011 12.1477C8.22784 12.298 8.61197 12.3754 9 12.3754C9.38803 12.3754 9.77216 12.298 10.1299 12.1477C10.4876 11.9974 10.8118 11.7772 11.0833 11.5M1.5 9C1.5 9.98491 1.69399 10.9602 2.0709 11.8701C2.44781 12.7801 3.00026 13.6069 3.6967 14.3033C4.39314 14.9997 5.21993 15.5522 6.12987 15.9291C7.03982 16.306 8.01509 16.5 9 16.5C9.98491 16.5 10.9602 16.306 11.8701 15.9291C12.7801 15.5522 13.6069 14.9997 14.3033 14.3033C14.9997 13.6069 15.5522 12.7801 15.9291 11.8701C16.306 10.9602 16.5 9.98491 16.5 9C16.5 8.01509 16.306 7.03982 15.9291 6.12987C15.5522 5.21993 14.9997 4.39314 14.3033 3.6967C13.6069 3.00026 12.7801 2.44781 11.8701 2.0709C10.9602 1.69399 9.98491 1.5 9 1.5C8.01509 1.5 7.03982 1.69399 6.12987 2.0709C5.21993 2.44781 4.39314 3.00026 3.6967 3.6967C3.00026 4.39314 2.44781 5.21993 2.0709 6.12987C1.69399 7.03982 1.5 8.01509 1.5 9Z"/></svg>');
			editor.ui.registry.addIcon('magicIconCover', '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M13 0.5C13.6377 0.499964 14.2513 0.743604 14.7152 1.18107C15.1792 1.61854 15.4584 2.21676 15.4958 2.85333L15.5 3V13C15.5 13.6377 15.2564 14.2513 14.8189 14.7152C14.3815 15.1792 13.7832 15.4584 13.1467 15.4958L13 15.5H3C2.36232 15.5 1.74874 15.2564 1.28478 14.8189C0.820828 14.3815 0.541577 13.7832 0.504167 13.1467L0.5 13V3C0.499964 2.36232 0.743604 1.74874 1.18107 1.28478C1.61854 0.820828 2.21676 0.541577 2.85333 0.504167L3 0.5H13ZM13.8333 5.5H2.16667V13C2.16669 13.2041 2.24163 13.4011 2.37726 13.5536C2.5129 13.7062 2.69979 13.8036 2.9025 13.8275L3 13.8333H13C13.2041 13.8333 13.4011 13.7584 13.5536 13.6227C13.7062 13.4871 13.8036 13.3002 13.8275 13.0975L13.8333 13V5.5Z"/></svg>');
			// editor.ui.registry.addIcon('magicIconTrash', '<svg width="16" height="18" viewBox="0 0 16 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1.33333 4.83333H14.6667M6.33333 8.16667V13.1667M9.66667 8.16667V13.1667M2.16667 4.83333L3 14.8333C3 15.2754 3.17559 15.6993 3.48815 16.0118C3.80072 16.3244 4.22464 16.5 4.66667 16.5H11.3333C11.7754 16.5 12.1993 16.3244 12.5118 16.0118C12.8244 15.6993 13 15.2754 13 14.8333L13.8333 4.83333M5.5 4.83333V2.33333C5.5 2.11232 5.5878 1.90036 5.74408 1.74408C5.90036 1.5878 6.11232 1.5 6.33333 1.5H9.66667C9.88768 1.5 10.0996 1.5878 10.2559 1.74408C10.4122 1.90036 10.5 2.11232 10.5 2.33333V4.83333" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>');
			// editor.ui.registry.addIcon('magicIconArrowUp', '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 5l0 14" /><path d="M18 11l-6 -6" /><path d="M6 11l6 -6" /></svg>');
			// editor.ui.registry.addIcon('magicIconArrowDown', '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 5l0 14" /><path d="M18 13l-6 6" /><path d="M6 13l6 6" /></svg>');
			// editor.ui.registry.addIcon('magicIconPencil', '<svg width="13" height="11" viewBox="0 0 13 11" fill="none" stroke="currentColor" stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"><path d="M8.23529 1.82353L10.4706 3.94118M1.52941 10.2941H4.15429L11.0446 3.76645C11.3927 3.43669 11.5882 2.98944 11.5882 2.52309C11.5882 2.05674 11.3927 1.60949 11.0446 1.27972C10.6965 0.949963 10.2244 0.764706 9.73216 0.764706C9.2399 0.764706 8.7678 0.949963 8.41972 1.27972L1.52941 7.80739V10.2941Z"/></svg>');

			editor.ui.registry.addButton( 'magicIconRewrite', {
				icon: 'magicIconRewrite',
				text: magicai_localize?.rewrite || 'Rewrite',
				onAction: menuItems.rewrite.onAction
			} );

			editor.ui.registry.addMenuButton( 'magicAIButton', {
				icon: 'magicIcon',
				fetch: callback => {
					const items = Object.values( menuItems ).splice( 1 ).map( val => ( { type: 'menuitem', ...val } ) );
					callback( items );
				}
			} );

			const nestedMenuItems = [
				'translateto'
			];

			Object.entries( menuItems ).forEach( ( [ key, val ] ) => {
				const addMethod = nestedMenuItems.find(item => item === key) ? 'addNestedMenuItem' : 'addMenuItem';
				editor.ui.registry[addMethod]( key, val );
			} );

			if (isGeneratorV2) {
				const emojiPickerWrapper = document.querySelector('.lqd-emoji-picker');

				editor.on('init', function (event) {

					// adding styles for emoji and cover
					editor.dom.addStyle(`
html { overflow-x: hidden; }
body { padding: 3rem 2rem 0; }
#lqd-editor-emoji-el[data-mce-selected], #lqd-editor-cover-el[data-mce-selected] { outline: none; }
#lqd-editor-emoji-el { position: relative; font-size: 55px; line-height: 1.25em; cursor: pointer; margin-bottom: 2rem; }
#lqd-editor-emoji-el-span { pointer-events: none; }
#lqd-editor-cover-el { display: flex; position: relative; margin-top: 1.5rem; margin-bottom: 1.5rem; }
#lqd-editor-cover-el:first-child, [data-mce-caret]:first-child + #lqd-editor-cover-el { margin-top: -3rem; }
#lqd-editor-cover-img { width: 100%; height: 210px; object-fit: cover; object-position: center; max-width: 100%; }
#lqd-editor-cover-el + #lqd-editor-emoji-el, #lqd-editor-cover-el + [data-mce-caret] + #lqd-editor-emoji-el { margin-top: -3.5rem; }

@media(min-width: 992px) {
	body {
		--content-width: 840px;
		width: var(--content-width);
		margin: 0 auto !important;
	}
	#lqd-editor-cover-el {
		width: 100vw;
		margin-inline-start: calc((50vw - var(--content-width) / 2) * -1);
	}
}
`);

					// actions we need when the content is changed
					const observer = new MutationObserver((mutationList, observer) => {
						const coverEl = editor.dom.get('lqd-editor-cover-el');

						editor.formElement.classList.toggle('has-cover', coverEl && editor.dom.nodeIndex(coverEl) === 0);
					});
					observer.observe(editor.getBody(), { childList: true, subtree: true });

					// event listeners
					const handleElementHover = (event, isHovering) => {
						const hoveredElement = event.target;
						const emojiElement = hoveredElement?.closest('#lqd-editor-emoji-el');
						const coverElement = hoveredElement?.closest('#lqd-editor-cover-el');

						if (emojiElement || coverElement) {
							const rect = editor.dom.getRect(hoveredElement);
							const customEvent = new CustomEvent('hovered-element', {
								detail: {
									element: emojiElement || coverElement,
									rect: rect,
									eventType: isHovering ? 'mouseover' : 'mouseout'
								},
								bubbles: true
							});
							editor.formElement.dispatchEvent(customEvent);
						}
					};

					editor.on('mouseover', event => {
						handleElementHover(event, true);
					});

					editor.on('mouseout', event => {
						handleElementHover(event, false);
					});

					editor.on('ScrollContent', event => {
						const customEvent = new CustomEvent('editor-scroll', {
							detail: {
								event: event.currentTarget
							},
							bubbles: true
						});
						editor.formElement.dispatchEvent(customEvent);
					});
				});

				if ( emojiPickerWrapper ) {
					const emojiPicker = picmo.createPicker({
						rootElement: emojiPickerWrapper
					});

					const positionEmojiPicker = targetElement => {
						try {
							const editorIframe = editor.iframeElement;
							const editorContainer = editor.container;
							const rect = editor.dom.getRect(targetElement);
							const windowHeight = window.innerHeight;
							const windowWidth = window.innerWidth;
							const pickerHeight = emojiPickerWrapper.offsetHeight || 400; // fallback height
							const pickerWidth = emojiPickerWrapper.offsetWidth || 350; // fallback width

							// Calculate initial position
							let top = editorContainer.offsetTop + rect.y + rect.h - (editorIframe.contentWindow.scrollY || 0) + 10;
							let left = targetElement.offsetLeft + editor.container.offsetLeft;

							// Adjust if picker would go below viewport
							if (top + pickerHeight > windowHeight) {
								top = Math.max(10, windowHeight - pickerHeight - 10);
							}

							// Adjust if picker would go outside right edge
							if (left + pickerWidth > windowWidth) {
								left = Math.max(10, windowWidth - pickerWidth - 10);
							}

							// Ensure picker doesn't go above viewport
							if (top < 0) {
								top = 10;
							}

							// Ensure picker doesn't go outside left edge
							if (left < 0) {
								left = 10;
							}

							emojiPickerWrapper.style.top = `${top}px`;
							emojiPickerWrapper.style.left = `${left}px`;
						} catch (error) {
							console.warn('Error positioning emoji picker:', error);
							// Fallback positioning
							emojiPickerWrapper.style.top = '50px';
							emojiPickerWrapper.style.left = '50px';
						}
					};

					const toggleEmojiPicker = (show = false, targetElement = null) => {
						if (show && targetElement) {
							positionEmojiPicker(targetElement);
							emojiPickerWrapper.classList.add('active');
						} else {
							emojiPickerWrapper.classList.remove('active');
						}
					};

					const getEmojiEl = () => {
						const editorBody = tinymce.activeEditor.getBody();
						const coverEl = editor.dom.get('lqd-editor-cover-el');
						let emojiEl = editor.dom.get('lqd-editor-emoji-el');

						if ( !emojiEl ) {
							emojiEl = editor.dom.create('p', {
								id: 'lqd-editor-emoji-el',
								contenteditable: 'false'
							});

							editor.dom.add(emojiEl, 'span', {
								id: 'lqd-editor-emoji-el-span',
								contenteditable: 'false'
							}, '💬');

							if (coverEl && editor.dom.nodeIndex(coverEl) === 0) {
								editor.dom.insertAfter(emojiEl, coverEl);
							} else if (editorBody.firstChild) {
								editorBody.insertBefore(emojiEl, editorBody.firstChild);
							} else {
								editorBody.appendChild(emojiEl);
							}
						}

						return emojiEl;
					};

					document.addEventListener('click', event => {
						if (!emojiPickerWrapper.contains(event.target) && !event.target.closest('#lqd-editor-emoji-el')) {
							toggleEmojiPicker(false);
						}
					});

					editor.on('click', event => {
						const clickedElement = event.target;
						const isEmojiElement = clickedElement?.id === 'lqd-editor-emoji-el';

						if (isEmojiElement) {
							toggleEmojiPicker(true, clickedElement);
						} else {
							toggleEmojiPicker(false);
						}
					});

					emojiPicker.addEventListener('emoji:select', event => {
						editor.dom.setHTML('lqd-editor-emoji-el-span', event.emoji);
						toggleEmojiPicker(false);
					});

					editor.ui.registry.addButton( 'magicAIEmoji', {
						text: magicai_localize?.add_emjoi ?? 'Add Emoji',
						icon: 'magicIconEmoji',
						onAction: function() {
							let emojiEl = getEmojiEl();

							toggleEmojiPicker(true, emojiEl);
						}
					} );
				}

				const onMediaSelect = event => {
					const { selectedItems } = event.detail;

					if ( !selectedItems.length ) return;

					const imageUrl = selectedItems[0]?.url;

					if ( !imageUrl ) return;

					const editorBody = tinymce.activeEditor.getBody();
					let coverEl = editor.dom.get('lqd-editor-cover-el');
					let coverImage = editor.dom.get('lqd-editor-cover-img');

					if ( !coverEl ) {
						coverEl = editor.dom.create('p', {
							id: 'lqd-editor-cover-el',
							contenteditable: 'false'
						});

						if (editorBody.firstChild) {
							editorBody.insertBefore(coverEl, editorBody.firstChild);
						} else {
							editorBody.appendChild(coverEl);
						}
					}

					if ( !coverImage ) {
						coverImage = editor.dom.create('img', {
							id: 'lqd-editor-cover-img',
							contenteditable: 'false'
						});

						editor.dom.add(coverEl, coverImage);
					}

					editor.dom.setAttrib(coverImage, 'src', imageUrl);
				};

				editor.ui.registry.addButton( 'magicAICover', {
					text: magicai_localize?.add_cover ?? 'Add Cover',
					icon: 'magicIconCover',
					onAction: function() {
						const input = document.querySelector('.editor-cover-input');

						if ( !input ) return;

						input.removeEventListener('mediaManagerSelection', onMediaSelect);
						input.addEventListener('mediaManagerSelection', onMediaSelect);

						input.click();
					}
				} );
			}

			editor.on( 'ContextMenu', function () {
				$('.tox-collection').remove();
				setTimeout( () => {
					$( '.tox-collection' ).css( 'width', 'clamp(200px, 320px, 90vw)' );

					$( '.tox-collection__item-label' ).eq(0).html( '<input id="custom_prompt" type="text" style="width: 100%!important" placeholder="What would you like to do?">' );

					$( '.tox-collection__group' ).eq(0).find('#custom_label').remove();
					$( '.tox-collection__group' ).eq(1).find( '#quick_label' ).remove();
					$( '.tox-collection__group' ).eq(0).prepend( '<p class="tox-custom-label" id="custom_label">CUSTOM ACTION</p>' );
					$( '.tox-collection__group' ).eq(1).prepend( '<p class="tox-custom-label" id="quick_label">QUICK ACTIONS</p>' );
				}, 0 );
			} );

			liquidTinyMCEThemeHandlerInit(editor);
		}
	};
}

( () => {
	'use strict';
	document.addEventListener( 'DOMContentLoaded', function () {
		document.querySelectorAll('.tinymce').forEach(el => tinyMCE.init( getTinyMCEOptions(el) ));

		$( 'body' ).on( 'click', '#workbook_regenerate', () => {
			sendOpenaiGeneratorForm();
		} );

		$( 'body' ).on( 'click', '#workbook_undo', () => {
			tinymce.activeEditor.execCommand( 'Undo' );
		} );
		$( 'body' ).on( 'click', '#workbook_redo', () => {
			tinymce.activeEditor.execCommand( 'Redo' );
		} );
		$( 'body' ).on( 'click', '#workbook_copy', () => {
			if ( window.codeRaw ) {
				navigator.clipboard.writeText( window.codeRaw );
				toastr.success( 'Code copied to clipboard' );
				return;
			}
			if ( tinymce?.activeEditor ) {
				tinymce.activeEditor.execCommand( 'selectAll', true );
				const content = tinymce.activeEditor.selection.getContent( { format: 'html' } );
				// Create a ClipboardItem for HTML
				const blob = new Blob([ content ], { type: 'text/html' });
				const item = new ClipboardItem({ 'text/html': blob });
				navigator.clipboard.write([ item ]).then(() => {
					toastr.success( 'Content copied to clipboard' );
				}).catch(err => {
					console.error('Failed to copy: ', err);
				});
				return;
			}
		} );
		$( 'body' ).on( 'click', '.workbook_download', event => {
			const button = event.currentTarget;
			const docType = button.dataset.docType;
			const docName = button.dataset.docName || 'document';

			tinymce.activeEditor.execCommand( 'selectAll', true );
			let content = tinymce.activeEditor.selection.getContent( { format: 'html' } );

			// Fix relative URLs
			const baseUrl = window.location.origin;
			const parser = new DOMParser();
			const doc = parser.parseFromString(content, 'text/html');

			const tags = [ 'a', 'img', 'video', 'audio', 'source' ];
			tags.forEach(tag => {
				doc.querySelectorAll(tag).forEach(element => {
					const attr = tag === 'a' ? 'href' : 'src';
					const url = element.getAttribute(attr);
					if (url && !url.startsWith('http')) {
						element.setAttribute(attr, new URL(url, baseUrl).href);
					}
				});
			});

			content = doc.documentElement.outerHTML;

			if ( docType === 'pdf' ) {
				const contentWithMargin = `<div style="margin: 20px;">${ content }</div>`;
				return html2pdf()
					.set({
						filename: docName
					})
					.from(contentWithMargin)
					.toPdf()
					.save();
			}

			const html = `
			<html ${ this.doctype === 'doc' ? 'xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40"' : '' }>
			<head>
				<meta charset="utf-8" />
				<title>${ docName }</title>
			</head>
			<body style="margin: 20px;">
				${ content }
			</body>
			</html>`;

			const url = `${ docType === 'doc' ? 'data:application/vnd.ms-word;charset=utf-8' : 'data:text/plain;charset=utf-8' },${ encodeURIComponent( html ) }`;

			const downloadLink = document.createElement( 'a' );
			document.body.appendChild( downloadLink );
			downloadLink.href = url;
			downloadLink.download = `${ docName }.${ docType }`;
			downloadLink.click();

			document.body.removeChild( downloadLink );

		} );
	} );

	if (stream_type == 'backend') {
		window.addEventListener('beforeunload', function(e) {
			$.ajax({
				type: 'post',
				url: '/dashboard/user/generator/reduce-tokens/wrtier',
				data: {
					streamed_text: streamed_text,
					streamed_message_id: streamed_message_id
				}
			});
		});
	}
} )();
function getResult() {
	'use strict';
	document.querySelectorAll('.tinymce').forEach(el => tinyMCE.init( getTinyMCEOptions(el) ));
}
function editWorkbook( workbook_slug ) {
	'use strict';

	document.querySelector( '.workbook-form' )?.classList?.add('loading');
	document.querySelector( '#workbook_button' ).disabled = true;
	Alpine.store('appLoadingIndicator').show();
	tinyMCE.get( 'workbook_text' ).save();
	var formData = new FormData();
	formData.append( 'workbook_slug', workbook_slug );
	formData.append( 'workbook_text', $( '#workbook_text' ).val() );
	formData.append( 'workbook_title', $( '#workbook_title' ).val() );

	$.ajax( {
		type: 'post',
		url: '/dashboard/user/openai/documents/workbook-save',
		data: formData,
		contentType: false,
		processData: false,
		success: function ( data ) {
			toastr.success( 'Workbook Saved Succesfully' );
		},
		error: function ( data ) {
			var err = data.responseJSON.errors;
			$.each( err, function ( index, value ) {
				toastr.error( value );
			} );
		},
		complete: function() {
			document.querySelector( '.workbook-form' )?.classList?.remove('loading');
			document.querySelector( '#workbook_button' ).disabled = false;
			Alpine.store('appLoadingIndicator').hide();
		}
	} );
	return false;
}

function endResponse(submitBtn, workbook_regenerate, typingEl) {
	submitBtn.classList.remove( 'lqd-form-submitting' );
	Alpine.store('appLoadingIndicator').hide();
	workbook_regenerate?.classList?.remove( 'hidden' );
	typingEl?.classList?.add( 'lqd-is-hidden' );
	submitBtn.disabled = false;
}
