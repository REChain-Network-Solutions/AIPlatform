jQuery(document).ready(function($) {
    "use strict";

    var notyf = new Notyf({
        duration: 5000,
        position: {
            x: 'right',
            y: 'bottom',
          },
          types: [
            {
              type: 'info',
              background: 'blue',
              icon: false
            }
          ]
    });

    var post_id = $("#post_ID").val();

    $(document).on("click", ".btn-fetch", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }
        let $this = $(this);
        let type = $this.attr('data-type');
        let url = $this.prev('#url').val();

        if ( !type ){
            notyf.error('Invalid action!');
            return false;
        }

        if ( !url ){
            notyf.error('Enter a website URL');
            return false;
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_' + type);
        form_data.append('url', url);
        form_data.append('post_id', post_id);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $this.addClass('is-loading');
            },
			success: function ( response ) {
                $this.removeClass('is-loading');
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.output ) {
                    $('.magicai-chatbot--train-url .result').remove();
                    $('.magicai-chatbot--train-url').append(response.output);
                    notyf.success('Fetched!');
                }
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

        
    });

    $(document).on("click", ".magicai-chatbot--train-url--delete", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let url = $this.attr('data-url');
        let is_trained = $this.attr('data-trained');

        if ( !url ){
            notyf.error('Something went wrong!');
            return false;
        }

        if ( is_trained == 'yes' ) {
            if ( !confirm('Trained data will be deleted. Are you sure?') ){
                return false;
            }
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_chatbot_delete_single_url');
        form_data.append('url', url);
        form_data.append('post_id', post_id);
        form_data.append('is_trained', is_trained);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
            },
			success: function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.message ) {
                    $this.parents().eq(2).remove();
                    notyf.success(response.message);
                }
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

        
    });

    $(document).on("click", ".magicai-chatbot--train-url--train", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);

        if ( !url ){
            notyf.error('Something went wrong!');
            return false;
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_chatbot_train_with_url');
        form_data.append('url', url);
        form_data.append('post_id', post_id);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $this.addClass('is-loading');
                $this.attr('disabled', true);
            },
			success: function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.message ) {
                    $('.magicai-chatbot--train-url .result').remove();
                    $('.magicai-chatbot--train-url').append(response.output);
                    notyf.success(response.message);
                }
                $this.removeClass('is-loading');
                $this.removeAttr('disabled');
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-chatbot--train-pdf-file", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        e.preventDefault();

        // Declare a variable for the media upload window
        var mediaUploader;
        var $this = $(this);

        // Open the media upload window
        if (mediaUploader) {
            mediaUploader.open();
            return;
        }

        // Create a new media upload window
        mediaUploader = wp.media({
            title: 'Select PDF',
            button: {
                text: 'Select PDF'
            },
            multiple: false, // Set to 'false' if you want to select a single media item
            library: {
                type: ['application/pdf'] // Specify your custom MIME types here
            }
        });

        // Action when media is selected
        mediaUploader.on('select', function() {
            var attachment = mediaUploader.state().get('selection').first().toJSON();

            var form_data = new FormData();
            form_data.append('action', 'magicai_chatbot_train_with_pdf');
            form_data.append('attachment_id', attachment.id);
            form_data.append('post_id', post_id);

            $.ajax({
                url: magicai_js_options.ajaxurl,
                type: 'POST',
                contentType: false,
                processData: false,
                data: form_data,
                beforeSend: function() {
                    $this.addClass('is-loading');
                    $this.attr('disabled', true);
                },
                success: function ( response ) {
                    $this.removeClass('is-loading');
                    $this.removeAttr('disabled');
                    if ( response.error ) {
                        notyf.error(response.message);
                        return false;
                    }
                    if ( response.message ) {
                        $('.magicai-chatbot--train-pdf .result').remove();
                        $('.magicai-chatbot--train-pdf').append(response.output);
                        notyf.success(response.message);
                    }
                },
                error: function ( response ) {
                    notyf.error(response.data);
                    
                }
            });

        });

        // Open the media upload window
        mediaUploader.open();
    });

    $(document).on("click", ".magicai-chatbot--train-pdf--delete", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let attachment_id = $this.attr('data-id');
        let is_trained = $this.attr('data-trained');

        if ( is_trained == 'yes' ) {
            if ( !confirm('Trained data will be deleted. Are you sure?') ){
                return false;
            }
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_chatbot_delete_pdf');
        form_data.append('attachment_id', attachment_id);
        form_data.append('post_id', post_id);
        form_data.append('is_trained', is_trained);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
            },
			success: function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.message ) {
                    $this.parents().eq(2).remove();
                    notyf.success(response.message);
                }
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-chatbot--train-text-btn, .magicai-chatbot--train-text--edit", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        var text_id = $(this).attr('data-id') ?? '';

        var currentModal = jQuery.confirm( {
            columnClass: 'magicai-modal',
            title: magicai_js_options.modal.title.chatbot_add_text,
            content: function(){
                var self = this;
                jQuery.post(
                    magicai_js_options.ajaxurl, 
                    { action: 'magicai_chatbot_get_train_text', post_id: post_id, text_id: text_id }, 
                    function ( response ) {
                        self.setContent(magicai_js_options.modal.content.chatbot_add_text);
                        $('.magicai-modal').find('input[name*="title"]').val(response.title);
                        $('.magicai-modal').find('textarea[name*="content"]').val(response.content);
                    }
                );
            },
            closeIcon: true,
            closeIconClass: 'dashicons dashicons-no',
            buttons: {
                new: {
                    btnClass: 'btn-blue',
                    text: 'Save',
                    action: function () {
                        
                        $('.magicai-modal').addClass('is-loading');

                        var title = this.$content.find('input[name*="title"]').val();
                        var content = this.$content.find('textarea[name*="content"]').val();
                        if(!title){
                            jQuery.alert('Enter a text title.');
                            return false;
                        }
                        if(!content){
                            jQuery.alert('Enter a text content.');
                            return false;
                        }
                        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_train_with_text', post_id: post_id, title: title, content: content, text_id: text_id }, function ( response ) {
                            if ( response.error ) {
                                notyf.error(response.message);
                            }
                            if ( response.message ) {
                                $('.magicai-chatbot--train-text .result').remove();
                                $('.magicai-chatbot--train-text').append(response.output);
                                notyf.success(response.message);
                                currentModal.close();
                            }
                            $('.magicai-modal').removeClass('is-loading');

                        });

                        return false;
                       
                    }
                },
            }
        } );

        e.preventDefault();
       
    });

    $(document).on("click", ".magicai-chatbot--train-text--delete", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let text_id = $this.attr('data-id');
        let is_trained = $this.attr('data-trained');

        if ( is_trained == 'yes' ) {
            if ( !confirm('Trained data will be deleted. Are you sure?') ){
                return false;
            }
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_chatbot_delete_text');
        form_data.append('text_id', text_id);
        form_data.append('post_id', post_id);
        form_data.append('is_trained', is_trained);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
            },
			success: function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.message ) {
                    $this.parents().eq(2).remove();
                    notyf.success(response.message);
                }
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-chatbot--train-qa-btn, .magicai-chatbot--train-qa--edit", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        var qa_id = $(this).attr('data-id') ?? '';

        var currentModal = jQuery.confirm( {
            columnClass: 'magicai-modal',
            title: magicai_js_options.modal.title.chatbot_add_qa,
            content: function(){
                var self = this;
                jQuery.post(
                    magicai_js_options.ajaxurl, 
                    { action: 'magicai_chatbot_get_train_qa', post_id: post_id, qa_id: qa_id }, 
                    function ( response ) {
                        self.setContent(magicai_js_options.modal.content.chatbot_add_qa);
                        $('.magicai-modal').find('input[name*="q"]').val(response.q);
                        $('.magicai-modal').find('textarea[name*="a"]').val(response.a);
                    }
                );
            },
            closeIcon: true,
            closeIconClass: 'dashicons dashicons-no',
            buttons: {
                new: {
                    btnClass: 'btn-blue',
                    text: 'Save',
                    action: function () {
                        
                        $('.magicai-modal').addClass('is-loading');

                        var q = this.$content.find('input[name*="q"]').val();
                        var a = this.$content.find('textarea[name*="a"]').val();
                        if(!q){
                            jQuery.alert('Enter a Question.');
                            return false;
                        }
                        if(!a){
                            jQuery.alert('Enter a Answer.');
                            return false;
                        }
                        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_train_with_qa', post_id: post_id, q: q, a: a, qa_id: qa_id }, function ( response ) {
                            if ( response.error ) {
                                notyf.error(response.message);
                            }
                            if ( response.message ) {
                                $('.magicai-chatbot--train-qa .result').remove();
                                $('.magicai-chatbot--train-qa').append(response.output);
                                notyf.success(response.message);
                                currentModal.close();
                            }
                            $('.magicai-modal').removeClass('is-loading');

                        });

                        return false;
                       
                    }
                },
            }
        } );

        e.preventDefault();
       
    });

    $(document).on("click", ".magicai-chatbot--train-qa--delete", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let qa_id = $this.attr('data-id');
        let is_trained = $this.attr('data-trained');

        if ( is_trained == 'yes' ) {
            if ( !confirm('Trained data will be deleted. Are you sure?') ){
                return false;
            }
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_chatbot_delete_qa');
        form_data.append('qa_id', qa_id);
        form_data.append('post_id', post_id);
        form_data.append('is_trained', is_trained);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
            },
			success: function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.message ) {
                    $this.parents().eq(2).remove();
                    notyf.success(response.message);
                }
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai--ban-chatbot-ip", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let ip = $this.attr('data-ip');

        var form_data = new FormData();
        form_data.append('action', 'magicai_chatbot_ban_ip');
        form_data.append('ip', ip);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
            },
			success: function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    return false;
                }
                if ( response.message ) {
                    $this.html(response.text);
                    notyf.success(response.message);
                }
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-custom-tab--nav-item", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }
        let nav = $(this);
        let target = nav.attr('data-href');
        let content = $(`.magicai-custom-tab--content[data-content="${target}"]`);
        

        $('.magicai-custom-tab--content,.magicai-custom-tab--nav-item').removeClass('active');
        content.addClass('active');
        nav.addClass('active');


        e.preventDefault();

    });

    $(document).on("click", "#magicai-chatbot--conversations .magicai-chatbot-widget--message-list--item", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }
        
        var chat_id = $(this).attr('data-id');
        var messages = $('#magicai-chatbot--conversations .magicai-chatbot-widget--messages');
        var messages_header = $('#magicai-chatbot--conversations .magicai-chatbot--conversations-message-actions');
        messages.empty();
        messages_header.empty();
        $('#magicai-chatbot--conversations').removeClass('init');
        $('#magicai-chatbot--conversations').addClass('is-loading');
        $('#magicai-chatbot--conversations .magicai-chatbot-widget--chat').toggleClass('active-message-list');
        $('#magicai-chatbot--conversations form').attr('data-id', chat_id);
        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_chatbot_get_chat_conversations', chat_id: chat_id }, function ( response ) {
            $('#magicai-chatbot--conversations').removeClass('is-loading');
            messages.append(response.output);
            messages_header.append(response.header);
            messages.scrollTop(messages[0].scrollHeight - messages[0].clientHeight);
        });

    });

});