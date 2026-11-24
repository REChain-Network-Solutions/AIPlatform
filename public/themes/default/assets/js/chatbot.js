jQuery(document).ready(function($) {
	'use strict';

	$(document).on('click', '#magicai-chatbot-widget .magicai-chatbot-widget--trigger', function (e) {
		if ( !magicai_js_options.guest_status ) {
			toastr.error(magicai_localize?.please_active_magicai ||
						'Please Active The MagicAI');
			return false;
		}

		var $this = $(this);
		var chat = $this.parent().find('.magicai-chatbot-widget--chat');
		var messages = chat.find('.magicai-chatbot-widget--messages');
		var message_list = chat.find('.magicai-chatbot-widget--message-list');
		chat.toggleClass('active');


		if ( ! $this.hasClass('loaded') ){
			$('#magicai-chatbot-widget').addClass('is-loading');
			messages.empty();
			jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_get_chat' }, function ( response ) {
				if ( response.chat_history ) {
					$('#magicai-chatbot-widget .magicai-chatbot-widget--chat').toggleClass('active-message-list');
					message_list.append(response.output);
				} else {
					$('#magicai-chatbot-widget form').attr('data-id', response.chat_id);
					messages.append(response.output);
				}
				$('#magicai-chatbot-widget').removeClass('is-loading');
				messages.scrollTop(messages[0].scrollHeight - messages[0].clientHeight);
				$this.addClass('loaded');
			});

		}
		e.preventDefault();

	});

	$(document).on('click', '#magicai-chatbot-widget .start-new-chat', function (e) {

		if ( !magicai_js_options.guest_status ) {
			toastr.error(magicai_localize?.please_active_magicai ||
						'Please Active The MagicAI');
			return false;
		}

		var messages = $('#magicai-chatbot-widget .magicai-chatbot-widget--messages');
		messages.empty();
		var message_list = $('#magicai-chatbot-widget .magicai-chatbot-widget--message-list');

		$('#magicai-chatbot-widget .magicai-chatbot-widget--chat').removeClass('active-message-list');
		$('#magicai-chatbot-widget').addClass('is-loading');

		jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_start_new_chat' }, function ( response ) {
			$('#magicai-chatbot-widget form').attr('data-id', response.chat_id);
			$('#magicai-chatbot-widget').removeClass('is-loading');
			messages.append(response.output);
			message_list.append(response.output_list);
			messages.scrollTop(messages[0].scrollHeight - messages[0].clientHeight);
		});

		e.preventDefault();

	});

	$(document).on('click', '#magicai-chatbot-widget .message-list', function (e) {

		if ( !magicai_js_options.guest_status ) {
			toastr.error(magicai_localize?.please_active_magicai ||
						'Please Active The MagicAI');
			return false;
		}
		$('#magicai-chatbot-widget .magicai-chatbot-widget--chat').toggleClass('active-message-list');

	});

	$(document).on('click', '#magicai-chatbot-widget .magicai-chatbot-widget--message-list--item', function (e) {

		if ( !magicai_js_options.guest_status ) {
			toastr.error(magicai_localize?.please_active_magicai ||
						'Please Active The MagicAI');
			return false;
		}

		var chat_id = $(this).attr('data-id');
		var messages = $('#magicai-chatbot-widget .magicai-chatbot-widget--messages');
		messages.empty();
		$('#magicai-chatbot-widget').addClass('is-loading');
		$('#magicai-chatbot-widget .magicai-chatbot-widget--chat').toggleClass('active-message-list');
		$('#magicai-chatbot-widget form').attr('data-id', chat_id);
		jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_get_chat', chat_id: chat_id }, function ( response ) {
			$('#magicai-chatbot-widget').removeClass('is-loading');
			messages.append(response.output);
			messages.scrollTop(messages[0].scrollHeight - messages[0].clientHeight);
		});

	});

	// send message
	let chatBotStop = false;
	$(document).on('submit', '#magicai-chatbot-widget form', function (e) {

		if ( !magicai_js_options.guest_status ) {
			toastr.error(magicai_localize?.please_active_magicai ||
						'Please Active The MagicAI');
			return false;
		}

		let form = $(this);
		form.addClass('is-working');
		let chat_id = form.attr('data-id');
		let result = $('#magicai-chatbot-widget .magicai-chatbot-widget--messages');
		let button = form.find('.start');
		// let button_stop = form.find('.stop');
		let prompt = form.find('#prompt').val();
		form.find('#prompt').val('');
		let message_id = Date.now();
		// var message_data = '';
		chatBotStop = false;

		result.append(`<div class="magicai-chatbot-widget--message">${prompt}</div>`);
		result.scrollTop(result[0].scrollHeight - result[0].clientHeight);

		jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_generate', prompt: prompt, chat_id: chat_id }, function ( response ) {
			if ( response.error ){
				toastr.error(response.message);
				result.append(`<div class="magicai-chatbot-widget--message ai text-${message_id}">${response.message}</div>`);
				result.scrollTop(result[0].scrollHeight - result[0].clientHeight);
				return false;
			} else {
				result.append(`<div class="magicai-chatbot-widget--message ai text-${message_id}"></div>`);
				result.scrollTop(result[0].scrollHeight - result[0].clientHeight);
				result.find( '.magicai-chatbot-widget--message.text-' + message_id ).text(response.message || '');
				form.removeClass('is-working');
			}
		});

		e.preventDefault();

	});

});
