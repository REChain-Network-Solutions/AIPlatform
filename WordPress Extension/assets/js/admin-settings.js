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

    var post_id, post_title, post_content, parentModal, post_tags, duplicate_url = '';

    // Check if the URL has a hash
    if (window.location.hash) {
        // Remove active class from all tabs
        $('.magicai-settings-form-page').removeClass('active');
        $('#magicai-settings-tabs-wrapper .nav-tab').removeClass('nav-tab-active');
        $('.magicai-page-description').removeClass('active');

        // Activate the tab based on the hash in the URL
        var targetTab = window.location.hash;
        $(targetTab).addClass('active');
        $(targetTab.replace('#tab-', '.')).addClass('active');
        $('[href="' + targetTab + '"]').addClass('nav-tab-active');
    } else {
        var targetTab = '#tab-custom-generator';
        $(targetTab).addClass('active');
        $(targetTab.replace('#tab-', '.')).addClass('active');
        $('[href="' + targetTab + '"]').addClass('nav-tab-active');
    }

    // Settings Tab
    $('#magicai-settings-tabs-wrapper').on('click', '.nav-tab', function() {
        $('.magicai-settings-form-page').removeClass('active');
        $('.magicai-page-description').removeClass('active');
        var targetTab = $(this).attr('href');
        $(targetTab).addClass('active');
        $(targetTab.replace('#tab-', '.')).addClass('active');
        $('#magicai-settings-tabs-wrapper .nav-tab').removeClass('nav-tab-active');
        $(this).addClass('nav-tab-active');
        $('html, body').animate({ scrollTop: 0 }, 100);
    });

    // Sortable
    let sortableItems = document.querySelectorAll('.magicai-settings--sortable-items');
    sortableItems.forEach(item => {
        Sortable.create(item, {
            animation: 150,
            handle: ".handle",
            onChange: function(){
                set_sortable_data();
            },
        });
    });

    // Sortable listen change
    $(document).on('input propertychange', '.magicai-settings--sortable-items input', function() {
        set_sortable_data();
    });

    // Sortable delete
    $(document).on('click', '.magicai-settings--sortable-item .remove', function() {
        $(this).closest('.magicai-settings--sortable-item').remove();
        set_sortable_data();
    });
    
    // Sortable Set data to settings
    function set_sortable_data() {

        var sortable = $('.magicai-settings--sortable-items');

        var menuData = [];
        sortable.children('.magicai-settings--sortable-item').each(function() {
            var obj = {}; 
            $(this).find("input").each(function() {
                let field = $(this).attr('data-field');
                let value = $(this).val();
                // console.log(value);
                obj[field] = value;
            });
        
            menuData.push(obj);
        });
        // console.log(menuData);
        
        sortable.next('.datas').val(JSON.stringify(menuData));
    }

    // Sortable - add field
    $(document).on("click", ".magicai-add--assistant-prompt", function (e) {
       
        $(this).next('.magicai-settings--sortable-items').
        append(`
        <div id="sortable-items" class="magicai-settings--sortable-items">
            <div class="magicai-settings--sortable-item">
                <div class="form-field">
                    <label for="icon">Icon (Dashicons or SVG)</label>
                    <input type="text" data-field="icon" name="icon" placeholder="Dashicons or SVG">
                </div>
                <div class="form-field w-100">
                    <label for="icon">Prompt</label>
                    <input type="text" data-field="prompt" name="prompt" placeholder="make shorter this paragraph">
                </div>
                <div class="action">
                    <div class="handle"><span class="dashicons dashicons-move"></span></div>
                    <div class="remove"><span class="dashicons dashicons-trash"></span></div>
                </div>
            </div>
        </div>`
        );
    });
    
    // Post Row Action
    $(document).on('click', '.magicai-row-action', function(e) {
        e.preventDefault();

        post_id = $(this).data('postid');
        duplicate_url = $(this).data('duplicate-url');

        jQuery.post( magicai_js_options.ajaxurl, { action: 'fetch_post_details', post_id: post_id }, function ( response ) {
            if ( response.success ) {
                post_title = response.data.title;
                post_content = response.data.content;
                post_tags = response.data.tags;
            } else {
                // error message
            }
        } );

        parentModal = jQuery.confirm( {
            columnClass: 'magicai-modal',
            title: magicai_js_options.modal.title.brand,
            content: magicai_js_options.modal.content.actions,
            closeIcon: true,
            closeIconClass: 'dashicons dashicons-no',
            buttons: {
                new: {
                    btnClass: 'btn-blue btn-hidden',
                    text: 'Generate →',
                    action: function () {
                    }
                },
            }
        } );
    });

    $(document).on("click", ".magicai-modal--actions span", function (e) {
        var prompt = $(this).data('prompt');
        var type = $(this).data('prompt-type');

        // duplicate post action
        if ( prompt == 'tools_duplicate' ) {
            window.location.href = duplicate_url;
            return;
        }

        // other actions
        var props = {
            post_id: post_id,
            post_title: post_title,
            post_content: post_content,
            post_tags: post_tags,
            type: type,
        };

        switch( type ) {
            case 'title':
                props.content = post_title;
                break;
            case 'content':
                props.content = post_content;
                break;
            case 'tag':
                if ( prompt == 'tag_generate_title' ) {
                    props.content = post_title;
                } else if ( prompt == 'tag_generate_content' ) {
                    props.content = post_content;
                } else if ( prompt == 'tag_add_more' ) {
                    props.content = post_tags ?? post_title;
                }
                break;
            case 'tools':
                props.content = 'tools';
                break;
        }

        magicai_create_request( prompt, props );
    });

    function magicai_create_request( prompt, props ) {

        jQuery('.magicai-modal').addClass('is-loading');

        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_block_editor', data: { prompt: prompt, content: props.content, return_esc_attr: 'yes' } }, function ( response ) {
            if ( response.error ) {
                notyf.error(response.message);
            } else {
                var currentModal = jQuery.confirm( {
                    columnClass: 'magicai-modal',
                    title: magicai_js_options.modal.title.brand,
                    content: `<div class="magicai-modal--results">
                                <p>Result:</p>
                                <div class="magicai-modal--prompt-wrapper">
                                    <input type="text" class="magicai-modal--result" value="${response.output}">
                                </div>
                            </div>`,
                    closeIcon: true,
                    closeIconClass: 'dashicons dashicons-no',
                    buttons: {
                        new: {
                            btnClass: 'btn-blue',
                            text: 'Apply to the Post →',
                            action: function () {
                                jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_update_post_list', data: { content: this.$content.find('.magicai-modal--result').val(), post_id: props.post_id, type: props.type } }, function ( response ) {
                                    parentModal.close();
                                    currentModal.close();
                                    location.reload();
                                    return false;
                                });
                            }
                        },
                    }
                } );
            }
            jQuery('.magicai-toolbar .dashicons-magicai-logo').removeClass('is-loading');
        } );
    }


    /**
     * Generators
     */

    $(document).on("submit", ".magicai-generator.post-generator", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');
        let image = form.find('#image').val();
        var form_data = new FormData();

        form_data.append('action', 'magicai_generate_post');
        form_data.append('title', form.find('#title').val());
        form_data.append('tag', form.find('#tag').val());
        form_data.append('language', form.find('#language').val());
        form_data.append('maximum_lenght', form.find('#maximum_lenght').val());
        form_data.append('number_of_results', form.find('#number_of_results').val());
        form_data.append('temperature', form.find('#temperature').val());
        form_data.append('tone', form.find('#tone').val());
        form_data.append('image', image);
        form_data.append('atts', form.find('#atts').val() ?? '');

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                    return;
                }

                if ( response.output ) {
                    notyf.success('Generated!');
                    result.empty();
                    result.removeClass('default');
                    result.append(response.output);
                    for ( let i = 0; i < response.n; i++ ) {
                        wp.editor.remove('post_content_' + i);
                        wp.editor.initialize('post_content_' + i, {tinymce: true,height: 300});
                        $( '#post_content_' +i+ '_ifr' ).css( 'height', '300px' );
                    }
                }
				
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-accordion-trigger", function (e) {
        $(this).toggleClass("active");
        var panel = $(this).parent().next();
        if (panel.css("display") === "block") {
          panel.css("display", "none");
        } else {
          panel.css("display", "block");
        }
    });
    
    $(document).on("click", ".magicai-accordion-action", function (e) {

        var $this = $(this);
        var post_id = $(this).data('postid');
        var action = $(this).data('action');
        var type = $(this).data('type');
        if ( post_id === '-1' ){ return; }
        var wrapper = $(this).closest('.generator-result');

        if ( action == 'copy-content' ) {
            var $element = wrapper.find( '#custom_content_' + post_id );
            $element.select();
            document.execCommand('copy')
            notyf.success( { message: 'Copied', duration: 1200 });
            return;
        }

        if ( action == 'copy-tinymce-content' ) {
            let editorContent = tinyMCE.get('custom_content_' + post_id).getContent();
            let tempTextarea = document.createElement('textarea');
            tempTextarea.value = editorContent;
            document.body.appendChild(tempTextarea);
            tempTextarea.select();
            document.execCommand('copy');
            document.body.removeChild(tempTextarea);
            notyf.success({ message: 'Copied', duration: 1200 });
            return;
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_add_new_post_as_draft');

        var post_title = wrapper.find( '#post_title_' + post_id ).val();
        if ( type == 'youtube' ) {
            var post_content = wp.editor.getContent( 'yt_post_content_' + post_id );
            var post_image = wrapper.find( '#post_image_' + post_id ).val();
            form_data.append('post_image', post_image);
        } else {
            var post_content = wp.editor.getContent( 'post_content_' + post_id );
            if ( type == 'product' ) {
                form_data.append('post_type', 'product');
                var post_content = wp.editor.getContent( 'product_content_' + post_id );
            }
        }
        var post_tags = wrapper.find( '#post_tags_' + post_id ).val() || '';
        var post_image = wrapper.find( 'input[name="generated-image"]:checked' ).val() || '';

        console.log('postimage:',post_image);

        form_data.append('post_title', post_title);
        form_data.append('post_content', post_content);
        form_data.append('post_tags', post_tags);
        form_data.append('post_image', post_image);

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
               $this.addClass('is-loading');
               $this.html('Processing...');
            },
			success: function ( response ) {
                $this.html(response.data.output);
                $this.removeClass('is-loading');
                $this.data('postid', '-1');
                notyf.success('Post saved as Draft!');
			},
			error: function ( response ) {
               notyf.error(response.data);
			}
		});

        e.preventDefault();

    });

    /**
     * Product Generator
     */

    $(document).on("submit", ".magicai-generator.product-generator", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');
        var form_data = new FormData();

        form_data.append('action', 'magicai_generate_product');
        form_data.append('title', form.find('#title').val());
        form_data.append('language', form.find('#language').val());
        form_data.append('maximum_lenght', form.find('#maximum_lenght').val());
        form_data.append('number_of_results', form.find('#number_of_results').val());
        form_data.append('temperature', form.find('#temperature').val());
        form_data.append('tone', form.find('#tone').val());
        form_data.append('atts', form.find('#atts').val() ?? '');

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                    return;
                }

                if ( response.output ) {
                    notyf.success('Generated!');
                    result.empty();
                    result.removeClass('default');
                    result.append(response.output);
                    for ( let i = 0; i < response.n; i++ ) {
                        wp.editor.remove('product_content_' + i);
                        wp.editor.initialize('product_content_' + i, {tinymce: true,height: 300});
                        $( '#product_content_' +i+ '_ifr' ).css( 'height', '300px' );
                    }
                }
				
			}
		});

        e.preventDefault();

    });

    /**
     * CODE
     */
    $(document).on("submit", ".magicai-generator.code-generator", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');
        var form_data = new FormData();

        form_data.append('action', 'magicai_generate_code');
        form_data.append('title', form.find('#title').val());
        form_data.append('code', form.find('#code').val());
        form_data.append('atts', form.find('#atts').val() ?? '');

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                    return;
                }

                if ( response.output ) {
                    notyf.success('Code Generated!');
                    result.empty();
                    result.removeClass('default');
                    result.append(response.output);
                    Prism.highlightAll();
                }
				
			}
		});

        e.preventDefault();

    });
    
    /**
     * Custom Generator
     */

    // CMD + ENTER
    $(document).on("keydown", ".magicai-generator.custom-generator", function (e) {
        if ((e.ctrlKey || e.metaKey) && (e.which == 13 || e.which == 10)) {
            $(this).closest(".magicai-generator.custom-generator").submit();
        }
    });

    $(document).on("submit", ".magicai-generator.custom-generator", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
            result.removeClass('default');
            result.find('.result-default').remove();
        let button = form.find('.btn');
        let prompt = form.find('#prompt').val();
        let title = prompt;
        let web_search = form.find('#web_search:checked').val() ? 1 : 0;
        let content_id = Date.now();
        
        $('body').addClass('magicai-loading');
        button.attr('disabled', true);
        if ( web_search ) {
            jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_web_search', prompt: prompt, web_search: web_search }, function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                    $('body').removeClass('magicai-loading');
                    button.removeAttr('disabled');
                    return;
                } else {
                    prompt = response.prompt;
                    asyncGenerate(prompt,title,result,button,content_id);
                }
            });
        } else {
            asyncGenerate(prompt,title,result,button,content_id);
        }
        e.preventDefault();
    
    });

    let asyncGenerate = async (prompt,title,result_wrapper,button,content_id) => {
        let chunk = [];
        let streaming = true;
        let controller = null; // Store the AbortController instance
        result_wrapper.prepend(`
            <button class="magicai-accordion">
                <div class="magicai-accordion-trigger active"></div>
                <div class="magicai-accordion-title">${title}</div>
                <div class="magicai-accordion-actions">
                    <div class="magicai-accordion-action" data-postid="${content_id}" data-action="copy-tinymce-content">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6ZM7 11H13V13H7V11ZM7 15H13V17H7V15Z"></path></svg>                        <span>Copy</span>
                    </div>
                </div>
            </button>
            <div class="magicai-accordion-panel" style="display:block">
                <div class="form-field">
                    <label for="custom_content_${content_id}">Result</label>
                    <textarea name="custom_content_${content_id}" id="custom_content_${content_id}" cols="30" rows="10" aria-hidden="true"></textarea>
                </div>
            </div>`
        );

        wp.editor.remove(`custom_content_${content_id}`);
        wp.editor.initialize(`custom_content_${content_id}`);
        
        controller = new AbortController();
        const signal = controller.signal;

        const response = await fetch(atob(magicai_js_options.guest_id), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: 'Bearer ' + atob(magicai_js_options.guest_event_id) + atob(magicai_js_options.guest_look_id) + atob(magicai_js_options.guest_product_id),
            },
            body: JSON.stringify({
                model: magicai_js_options.model.custom,
                messages: [{
                    role: 'user',
                    content: prompt,
                }],
                stream: true, // For streaming responses
            }),
            signal, // Pass the signal to the fetch request
        });
        
        if(response.status != 200) {
            const responseData = await response.json();
            notyf.error( responseData.error.message);
            $('body').removeClass('magicai-loading');
            button.removeAttr('disabled');
            throw response;
        }
        // Read the response as a stream of data
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let result = '';

        while (true) {
            // if ( window.console || window.console.firebug ) {
            // 	console.clear();
            // }
            const { done, value } = await reader.read();
            if (done) {
                streaming = false;
                break;
            }
            // Massage and parse the chunk of data
            const chunk1 = decoder.decode(value);
            const lines = chunk1.split("\n");
    
            const parsedLines = lines
                .map((line) => line.replace(/^data: /, "").trim()) // Remove the "data: " prefix
                .filter((line) => line !== "" && line !== "[DONE]") // Remove empty lines and "[DONE]"
                .map((line) => {
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
                // const { finish_reason } = choices[0];

                if (content) {
                    chunk.push(content);
                    result += content.replace( /(?:\r\n|\r|\n)/g, ' <br> ' );
			        tinyMCE.activeEditor.setContent( result, { format: 'raw' } );
                }
            }
        }
        $('body').removeClass('magicai-loading');
        button.removeAttr('disabled');
        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_add_new_documents', title: title, content: tinyMCE.activeEditor.getContent() }, function ( response ) {
            if ( response.error ) {
                notyf.error(response.message);
            } else {
                notyf.success(response.message);
            }
        });
    }    

    /**
     * 
     * IMAGE
     * 
     */
    $(document).on("click", ".magicai-generator.image-generator .suprise-me", function (e) {

        var prompt = $(this).parent().next('#prompt');

            const texts = [
                "A fire-breathing dragon wearing a top hat and sunglasses",
                "A serene beach at sunset with palm trees swaying in the breeze",
                "A futuristic cityscape with flying cars and holographic billboards",
                "A cozy library with floor-to-ceiling bookshelves and a fireplace",
                "A playful kitten riding a skateboard down a winding road",
                "An alien landscape with towering mushrooms and bioluminescent plants",
                "A steampunk-style airship sailing through the clouds",
                "A whimsical underwater world with talking fish and mermaids",
                "A grand castle atop a mountain with a cascading waterfall in the background",
                "A cyberpunk hacker in a neon-lit room surrounded by floating computer code",
            ];
          
            // Generate a random index within the array length
            const randomIndex = Math.floor(Math.random() * texts.length);
          
            // Return the random text
            prompt.val(texts[randomIndex]);
        
    });

    $(document).on("click", ".sd-upscale", function (e) {
        e.preventDefault();
        $(".magicai-generator.image-generator").data('upscale','upscale');
        $(".magicai-generator.image-generator").submit();
    });

    $(document).on("submit", ".magicai-generator.image-generator", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result .gallery');
        let button = form.find('.btn');
        let type = form.find( '.image-generator--types [name="type"]:checked' ).val() ?? 'dalle';
        let upscale = $(this).data('upscale');
        var form_data = new FormData();
        form_data.append('prompt', form.find('#prompt').val());
        form_data.append('image', form.find('#image').val());
        form_data.append('mask', form.find('#mask').val());
        form_data.append('variation', form.find('#variation').is(":checked") ? 1 : 0 );
        form_data.append('type', type );
        form_data.append('atts', form.find('#atts').val() ?? '');
        
        if ( type == 'dalle' ) {
            form_data.append('action', 'magicai_generate_image');
        } else {
            form_data.append('action', 'magicai_generate_sd_image');
            form_data.append('sd-image', form.find('#sd-image').val());
            form_data.append('upscale', upscale ? 1 : 0);
            $(this).removeData("upscale");
            
            form_data.append('style_preset', form.find('#style_preset').val());
            form_data.append('mood', form.find('#mood').val());
            form_data.append('sampler', form.find('#sampler').val());
            form_data.append('clip_guidance_preset', form.find('#clip_guidance_preset').val());
            form_data.append('seed', form.find('#seed').val());
            form_data.append('steps', form.find('#steps').val());
            form_data.append('image_resolution', form.find('#image_resolution').val());
            form_data.append('negative_prompt', form.find('#negative_prompt').val());

        }

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                    return;
                }

                if ( response.output ) {
                    notyf.success('Generated!');
                    result.prepend(response.output);
                }
				
			}
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-generator.image-generator .media-uploader", function (e) {
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
            title: 'Select Media',
            button: {
                text: 'Select Media'
            },
            multiple: false // Set to 'false' if you want to select a single media item
        });

        // Action when media is selected
        mediaUploader.on('select', function() {
            var attachment = mediaUploader.state().get('selection').first().toJSON();
            var mediaUrl = attachment.id;

            // Copy the URL of the selected media to the input field
            $this.next('input').val(mediaUrl);
            $this.find('.file-name').remove();
            var svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM13.4142 13.9997L15.182 15.7675L13.7678 17.1817L12 15.4139L10.2322 17.1817L8.81802 15.7675L10.5858 13.9997L8.81802 12.232L10.2322 10.8178L12 12.5855L13.7678 10.8178L15.182 12.232L13.4142 13.9997ZM9 4V6H15V4H9Z"></path></svg>';
            $this.append(`<span class="file-name">(${attachment.title})</span><span type="button" class="delete">${svg}</span>`);
            if ( $this.parent().hasClass('dall-e') ) {
                $('.magicai-generator.image-generator .variation-wrapper').css('display','block');
            }
            if ( $this.parent().hasClass('sd') ) {
                $('form.image-generator.sd .sd-upscale').css('display','inline-flex');
            }
        });

        // Open the media upload window
        mediaUploader.open();
    });

    $(document).on("click", ".magicai-generator.image-generator .media-uploader .delete", function (e) {
        e.stopPropagation();
        var $this = $(this);
        $this.prev('.file-name').remove();
        $this.parent().next('input').val("");
        $('.magicai-generator.image-generator .variation-wrapper').css('display','none');
        $('form.image-generator.sd .sd-upscale').css('display','none');
        $this.remove();
    });


    $(document).on("change", ".magicai-generator.image-generator #variation", function (e) {
        if ($(this).is(':checked')) {
            $('.magicai-generator.image-generator label[for="mask"]').css('display','none');
        } else {
            $('.magicai-generator.image-generator label[for="mask"]').css('display','block');
        }
    });
    
    $(document).on("click", ".magicai-generator.image-generator .image-generator--types label", function (e) {
        var type = $(this).prev('input').val();
        var form = $(this).closest('form');
        $('.magicai-generator.image-generator .image-generator--types label').removeClass('selected');
        $(this).addClass('selected');

        if ( type == 'dalle' ) {
            form.removeClass( 'sd' );
            $('.magicai-generator.image-generator .advanced-options-form').removeClass('active');
        } else {
            form.addClass( 'sd' );
        }
    });
    
    $(document).on("click", ".magicai-generator.image-generator .image-generator--sd-options label", function (e) {
        $('.magicai-generator.image-generator .image-generator--sd-options label').removeClass('selected');
        $(this).addClass('selected');
    });

    $(document).on("click", ".magicai-generator.image-generator .advanced-options-trigger", function (e) {
        $('.magicai-generator.image-generator .advanced-options-form').toggleClass('active');
    });

    // Gallery image modal
    $(document).on("click", "#tab-image-generator .gallery-item", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        e.preventDefault();

        var $this = $(this),
            id = $this.data('id'),
            src = $this.data('src'),
            name = $this.data('name'),
            prompt = $this.data('prompt');

        var download_svg = '<svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path d="M3 19H21V21H3V19ZM13 13.1716L19.0711 7.1005L20.4853 8.51472L12 17L3.51472 8.51472L4.92893 7.1005L11 13.1716V2H13V13.1716Z"></path></svg>';
        var trash_svg = '<svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM9 11H11V17H9V11ZM13 11H15V17H13V11ZM9 4V6H15V4H9Z"></path></svg>';
        var external_svg = '<svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M10 6V8H5V19H16V14H18V20C18 20.5523 17.5523 21 17 21H4C3.44772 21 3 20.5523 3 20V7C3 6.44772 3.44772 6 4 6H10ZM21 3V11H19L18.9999 6.413L11.2071 14.2071L9.79289 12.7929L17.5849 5H13V3H21Z"></path></svg>';

        jQuery.confirm( {
            boxWidth: '300px',
            columnClass: 'magicai-modal attachment-detail',
            backgroundDismiss: true,
            title: false,
            content: `<img src="${src}"><div class="detail"><p>${prompt}</p></div><a href="${src}" target="_blank" class="external">${external_svg}</a>`,
            buttons: {
                download: {
                    // btnClass: 'btn-blue',
                    text: download_svg + ' Download',
                    action: function () {
                        $.ajax({
                            url: src,
                            method: 'GET',
                            xhrFields: {
                                responseType: 'blob'
                            },
                            success: function (data) {
                                var a = document.createElement('a');
                                var url = window.URL.createObjectURL(data);
                                a.href = url;
                                a.download = name;
                                document.body.append(a);
                                a.click();
                                a.remove();
                                window.URL.revokeObjectURL(url);
                                notyf.success('Download started!');
                            }
                        });
                        return false;
                    }
                },
                delete: {
                    // btnClass: 'btn',
                    text: trash_svg + ' Delete',
                    action: function () {
                        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_delete_attachment', attachment_id: id }, function ( response ) {
                            notyf.success(response.data);
                        } );
                        $this.remove();
                    }
                },
            }
        } );

    });

    // CHAT
    let chatStop = false;
    $(document).on("submit", ".magicai-chat--form", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let form = $(this);
            form.addClass('is-working');
        let chat_type = form.attr('data-type');
        let image_data = form.attr('data-image');
        let pdf = {id: form.attr('data-id'), filename: form.attr('data-filename'), url: form.attr('data-url')};
        let result = $(this).parent().find('.magicai-chat--message-list');
        let chat_list = $(this).parent().parent().find('.magicai-chat--list');
        let post_id = result.attr('data-postid');
        let button = form.find('.start');
        let button_stop = form.find('.stop');
        let prompt = form.find('#prompt').val();
        let prompt_first = prompt;
            form.find('#prompt').val('');
        let web_search = form.parents().eq(2).find('#web_search:checked').val() ? 1 : 0;
        let message_id = Date.now();
        var message_data = '';
        chatStop = false;

        if ( chat_type == 'vision' && image_data ) {
            result.append(`
            <div class="magicai-chat--message">
                <div class="text">
                    <img width="32" src="${magicai_js_options.user_avatar}">
                    <div>${prompt}</div>
                </div>
            </div>
            <div class="magicai-chat--message image">
                <div class="text">
                    <img class="ai" src="${image_data}"/>
                </div>
            </div>`);
        } else if ( chat_type == 'pdf' && pdf.id ) {
            result.append(`
            <div class="magicai-chat--message">
                <div class="text">
                    <img width="32" src="${magicai_js_options.user_avatar}">
                    <div>${prompt}</div>
                </div>
            </div>
            <div class="magicai-chat--message image">
                <a class="text pdf" href="${pdf.url}" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="none"><path fill="#E9E9E0" d="M23.776 0H5.12c-.52 0-.94.421-.94 1.238v34.12c0 .22.42.642.94.642h25.762c.52 0 .94-.421.94-.643V8.343c0-.447-.06-.591-.165-.697l-7.48-7.48a.568.568 0 0 0-.4-.166Z"/><path fill="#D9D7CA" d="M24.107.097v7.617h7.618L24.107.097Z"/><path fill="#CC4B4C" d="M12.544 21.423c-.223 0-.438-.073-.621-.21-.67-.502-.76-1.06-.717-1.441.117-1.047 1.411-2.142 3.848-3.258.966-2.12 1.886-4.73 2.435-6.91-.642-1.397-1.265-3.209-.81-4.271.159-.373.357-.658.728-.782.147-.048.517-.11.653-.11.324 0 .609.417.81.674.19.242.62.754-.239 4.373.867 1.79 2.095 3.613 3.271 4.861.843-.152 1.568-.23 2.159-.23 1.006 0 1.616.235 1.865.718.206.4.122.867-.25 1.389-.358.5-.852.765-1.428.765-.781 0-1.692-.493-2.707-1.468a30.8 30.8 0 0 0-5.675 1.814c-.537 1.14-1.052 2.059-1.532 2.732-.659.923-1.227 1.354-1.79 1.354Zm1.712-3.296c-1.374.772-1.934 1.407-1.974 1.764-.007.06-.024.215.277.445.096-.03.655-.285 1.697-2.209Zm8.767-2.855c.523.403.651.607.994.607.15 0 .58-.007.778-.284.096-.134.133-.22.148-.267-.08-.041-.184-.126-.756-.126-.324 0-.733.014-1.165.07ZM18.22 11.04a45.826 45.826 0 0 1-1.719 4.863 32.123 32.123 0 0 1 4.176-1.299c-.867-1.007-1.735-2.266-2.457-3.563Zm-.39-5.44c-.063.022-.855 1.13.062 2.068.61-1.36-.034-2.076-.062-2.067ZM30.881 36H5.12a.94.94 0 0 1-.94-.94V25.07h27.643v9.988a.94.94 0 0 1-.94.941Z"/><path fill="#fff" d="M11.176 34.071h-1.055v-6.477h1.863c.275 0 .548.044.817.132.27.088.511.22.725.395.214.176.387.388.52.637.13.249.197.529.197.84 0 .328-.056.625-.167.892a1.866 1.866 0 0 1-.466.673c-.2.18-.44.322-.72.421-.282.1-.593.15-.932.15h-.783l.001 2.337Zm0-5.677v2.566h.967c.128 0 .256-.022.382-.066a.962.962 0 0 0 .348-.216c.105-.1.19-.238.254-.417a1.976 1.976 0 0 0 .053-1.028c-.03-.137-.09-.27-.18-.395a1.066 1.066 0 0 0-.383-.316c-.164-.085-.38-.128-.65-.128h-.791ZM20.712 30.653c0 .533-.057.988-.172 1.366a3.395 3.395 0 0 1-.435.95 2.236 2.236 0 0 1-.593.602c-.22.147-.432.256-.637.33-.205.073-.393.12-.563.14-.17.02-.295.03-.378.03h-2.452v-6.477h1.951c.546 0 1.025.087 1.437.26.413.171.756.402 1.029.689.272.287.476.614.61.98.136.366.203.743.203 1.13Zm-3.129 2.645c.715 0 1.23-.228 1.547-.685.316-.457.474-1.12.474-1.987 0-.269-.032-.536-.096-.8a1.711 1.711 0 0 0-.373-.716 1.971 1.971 0 0 0-.752-.518c-.316-.132-.726-.198-1.23-.198h-.616v4.904h1.046ZM23.314 28.394v2.039h2.707v.72h-2.707v2.918H22.24v-6.477h4.052v.8h-2.98Z"/></svg>
                    ${pdf.filename}
                </a>
            </div>`);
        } else {
            result.append(`
            <div class="magicai-chat--message">
                <div class="text">
                    <img width="32" src="${magicai_js_options.user_avatar}">
                    <div>${prompt}</div>
                </div>
            </div>`);
        }

        result.append(`
        <div class="magicai-chat--message ai">
            <div class="text text-${message_id}">
                <img width="32" src="${magicai_js_options.logo_url}">
                <div></div>
            </div>
            <div class="magicai-chat--message-action">
                <div class="btn copy">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6ZM7 11H13V13H7V11ZM7 15H13V17H7V15Z"></path></svg>
                    <span>Copy</span>
                </div>
            </div>
        </div>`);
        
        $('.magicai-chat--message-list').scrollTop($('.magicai-chat--message-list')[0].scrollHeight - $('.magicai-chat--message-list')[0].clientHeight);
        $('body').addClass('magicai-loading');
        if ( !post_id ) {
            jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_create_new_chat', type: chat_type }, function ( response ) {
                result.attr('data-postid', response.data );
                // console.log(chat_list);
                post_id = response.data;
                if (chat_list.find('.today').length > 0) {
                    chat_list.find('.today').after(`
                    <div class="magicai-chat--list-chat">
                        <div class="magicai-chat--list-chat--trigger" data-postid="${response.data}"></div>
                        <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M5.76282 17H20V5H4V18.3851L5.76282 17ZM6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455Z"></path></svg></div>
                        <div class="message">Chat #${response.data}</div>
                        <div class="dropdown">
                            <details>
                                <summary>
                                <div class="dropdown-trigger">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="18" height="18">
                                        <path d="M10 3a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM10 8.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM11.5 15.5a1.5 1.5 0 10-3 0 1.5 1.5 0 003 0z" />
                                    </svg>
                                </div>
                                </summary>
                                <div class="dropdown-content">
                                    <span class="magicai-chat--list-chat--action" data-action="edit" data-postid="${response.data}">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.41421 15.89L16.5563 5.74786L15.1421 4.33365L5 14.4758V15.89H6.41421ZM7.24264 17.89H3V13.6474L14.435 2.21233C14.8256 1.8218 15.4587 1.8218 15.8492 2.21233L18.6777 5.04075C19.0682 5.43128 19.0682 6.06444 18.6777 6.45497L7.24264 17.89ZM3 19.89H21V21.89H3V19.89Z"></path></svg>
                                        Edit
                                    </span>
                                    <span class="magicai-chat--list-chat--action" data-action="delete" data-postid="${response.data}">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M4 8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8ZM6 10V20H18V10H6ZM9 12H11V18H9V12ZM13 12H15V18H13V12ZM7 5V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V5H22V7H2V5H7ZM9 4V5H15V4H9Z"></path></svg>
                                        Delete
                                    </span>
                                </div>
                            </details>
                        </div>
                    </div>
                    `);
                } else {
                    chat_list.find('.magicai-chat--new').after('<div class="magicai-chat--list-date today">Today</div>');
                    chat_list.find('.today').after(`
                    <div class="magicai-chat--list-chat">
                        <div class="magicai-chat--list-chat--trigger" data-postid="${response.data}"></div>
                        <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M5.76282 17H20V5H4V18.3851L5.76282 17ZM6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455Z"></path></svg></div>
                        <div class="message">
                            Chat #${response.data}
                        </div>
                    </div>
                    `);
                }
                jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_get_chat_data', post_id:post_id }, function ( response ) {
                    message_data = response.data;
                    if ( web_search ) {
                        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_web_search', prompt: prompt, web_search: web_search }, function ( response ) {
                            if ( response.error ) {
                                notyf.error(response.message);
                                $('body').removeClass('magicai-loading');
                                button.removeAttr('disabled');
                                return;
                            } else {
                                prompt = response.prompt;
                                asyncChat(prompt_first, prompt,result,button,message_id,message_data,post_id,chat_type,image_data,pdf);
                            }
                        });
                    } else {
                        asyncChat(prompt_first, prompt,result,button,message_id,message_data,post_id,chat_type,image_data,pdf);
                    }
                });
            } );
        } else {
            jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_get_chat_data', post_id:post_id }, function ( response ) {
                message_data = response.data;
                if ( web_search ) {
                    jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_web_search', prompt: prompt, web_search: web_search }, function ( response ) {
                        if ( response.error ) {
                            notyf.error(response.message);
                            $('body').removeClass('magicai-loading');
                            button.removeAttr('disabled');
                            return;
                        } else {
                            prompt = response.prompt;
                            asyncChat(prompt_first,prompt,result,button,message_id,message_data,post_id,chat_type,image_data,pdf);
                        }
                    });
                } else {
                    asyncChat(prompt_first,prompt,result,button,message_id,message_data,post_id,chat_type,image_data,pdf);
                }
            });
        }

        form.removeAttr('data-image');
        form.removeAttr('data-id');
        form.removeAttr('data-filename');
        form.removeAttr('data-url');
        form.find('button.upload .badge').remove();

        e.preventDefault();
    
    });

    let asyncChat = async (prompt_first,prompt,result,button,message_id,message_data,post_id,chat_type,image_data,pdf) => {

        if ( pdf.id ) {
            notyf.success('The PDF file is currently being read. Please wait for a moment.');
            const embedded_data = await jQuery.post( 
                magicai_js_options.ajaxurl, 
                { action: 'magicai_pdf_parse', attachment_id: pdf.id, post_id: post_id }, 
                function ( response ) {}
            );
        }

        if ( chat_type == 'pdf' ) {
            const final_prompt = await jQuery.post( 
                magicai_js_options.ajaxurl, 
                { action: 'magicai_getMostSimilarText', post_id: post_id, prompt: prompt }, 
                function ( response ) {
                    if ( response.extra_prompt ) {
                        prompt = response.extra_prompt;
                    }
                }
            );
        }

        let chunk = [];
        let streaming = true;
        let controller = null; // Store the AbortController instance
        let message = result.find( '.magicai-chat--message .text-'+message_id+' div' );

        if ( chat_type == 'vision' && image_data ) {
            prompt =  [
                {type: "text", text: prompt},
                {
                    type: "image_url",
                    image_url: {
                        url: image_data
                    },
                    
                }
            ];
        }        

        controller = new AbortController();
        const signal = controller.signal;

        const response = await fetch(atob(magicai_js_options.guest_id), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: 'Bearer ' + atob(magicai_js_options.guest_event_id) + atob(magicai_js_options.guest_look_id) + atob(magicai_js_options.guest_product_id),
            },
            body: JSON.stringify({
                model: chat_type == 'vision' ? 'gpt-4-vision-preview' : magicai_js_options.model.chat,
                messages: message_data.concat([{'role': 'user', 'content': prompt }]),
                max_tokens: 2000,
                stream: true, // For streaming responses
            }),
            signal, // Pass the signal to the fetch request
        });
        
        if(response.status != 200) {
            const responseData = await response.json();
            notyf.error( responseData.error.message);
            $('.magicai-chat--form').removeClass('is-working');
            $('body').removeClass('magicai-loading');
            throw response;
        }
        // Read the response as a stream of data
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
            // if ( window.console || window.console.firebug ) {
            // 	console.clear();
            // }

            if (chatStop) {
                return; // Exit the function
            }
            const { done, value } = await reader.read();
            if (done) {
                streaming = false;
                break;
            }
            // Massage and parse the chunk of data
            const chunk1 = decoder.decode(value);
            const lines = chunk1.split("\n");
    
            const parsedLines = lines
                .map((line) => line.replace(/^data: /, "").trim()) // Remove the "data: " prefix
                .filter((line) => line !== "" && line !== "[DONE]") // Remove empty lines and "[DONE]"
                .map((line) => {
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
                // const { finish_reason } = choices[0];

                if (content) {
                    chunk.push(content);
                    message.append(content.replace( /(?:\r\n|\r|\n)/g, ' <br> ' ));
                }
            }
        }
        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_save_chat_data', post_id:post_id, prompt: prompt_first, message: chunk.join(""), pdf: pdf.id ?? null }, function ( response ) {
            $('body').removeClass('magicai-loading');
            $('.magicai-chat--message-list').scrollTop($('.magicai-chat--message-list')[0].scrollHeight - $('.magicai-chat--message-list')[0].clientHeight);
            $('.magicai-chat--form').removeClass('is-working');
        });
    }

    $(document).on("click", "#chat-form .stop", function (e) {
        chatStop = true;
        $('.magicai-chat--form').removeClass('is-working');
        $('body').removeClass('magicai-loading');
        notyf.success( { message: 'Generation stopped.', duration: 1200 } );
    });

    $(document).on("click", ".magicai-chat--new", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        var chat_type = $this.attr('data-type'); 
        var message_wrapper = $this.parents().eq(1).find('.magicai-chat--message-list');
        var chat_wrapper = $this.parents().eq(1).find('.magicai-chat--list');
            chat_wrapper.removeClass('toggle');
        var mobile_toggle = $this.parents().eq(1).find('.magicai-chat--toggle');
        mobile_toggle.removeClass('toggle');
        var header_title = $this.parents().eq(1).find('.magicai-chat--header-title');

        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_create_new_chat', type: chat_type }, function ( response ) {
            message_wrapper.empty();
            message_wrapper.attr('data-postid', response.data );
            header_title.html(`Chat #${response.data}`);

            if (chat_wrapper.find('.today').length > 0) {
                chat_wrapper.find('.today').after(`
                <div class="magicai-chat--list-chat">
                    <div class="magicai-chat--list-chat--trigger" data-postid="${response.data}"></div>
                    <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M5.76282 17H20V5H4V18.3851L5.76282 17ZM6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455Z"></path></svg></div>
                    <div class="message">Chat #${response.data}</div>
                    <div class="dropdown">
                        <details>
                            <summary>
                            <div class="dropdown-trigger">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="18" height="18">
                                    <path d="M10 3a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM10 8.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM11.5 15.5a1.5 1.5 0 10-3 0 1.5 1.5 0 003 0z" />
                                </svg>
                            </div>
                            </summary>
                            <div class="dropdown-content">
                                <span class="magicai-chat--list-chat--action" data-action="edit" data-postid="${response.data}">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.41421 15.89L16.5563 5.74786L15.1421 4.33365L5 14.4758V15.89H6.41421ZM7.24264 17.89H3V13.6474L14.435 2.21233C14.8256 1.8218 15.4587 1.8218 15.8492 2.21233L18.6777 5.04075C19.0682 5.43128 19.0682 6.06444 18.6777 6.45497L7.24264 17.89ZM3 19.89H21V21.89H3V19.89Z"></path></svg>
                                    Edit
                                </span>
                                <span class="magicai-chat--list-chat--action" data-action="delete" data-postid="${response.data}">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M4 8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8ZM6 10V20H18V10H6ZM9 12H11V18H9V12ZM13 12H15V18H13V12ZM7 5V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V5H22V7H2V5H7ZM9 4V5H15V4H9Z"></path></svg>
                                    Delete
                                </span>
                            </div>
                        </details>
                    </div>
                </div>
                `);
            } else {
                chat_wrapper.find('.magicai-chat--new').after('<div class="magicai-chat--list-date today">Today</div>');
                chat_wrapper.find('.today').after(`
                <div class="magicai-chat--list-chat">
                    <div class="magicai-chat--list-chat--trigger" data-postid="${response.data}"></div>
                    <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M5.76282 17H20V5H4V18.3851L5.76282 17ZM6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455Z"></path></svg></div>
                    <div class="message">Chat #${response.data}</div>
                    <div class="dropdown">
                        <details>
                            <summary>
                            <div class="dropdown-trigger">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="18" height="18">
                                    <path d="M10 3a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM10 8.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM11.5 15.5a1.5 1.5 0 10-3 0 1.5 1.5 0 003 0z" />
                                </svg>
                            </div>
                            </summary>
                            <div class="dropdown-content">
                                <span class="magicai-chat--list-chat--action" data-action="edit" data-postid="${response.data}">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.41421 15.89L16.5563 5.74786L15.1421 4.33365L5 14.4758V15.89H6.41421ZM7.24264 17.89H3V13.6474L14.435 2.21233C14.8256 1.8218 15.4587 1.8218 15.8492 2.21233L18.6777 5.04075C19.0682 5.43128 19.0682 6.06444 18.6777 6.45497L7.24264 17.89ZM3 19.89H21V21.89H3V19.89Z"></path></svg>
                                    Edit
                                </span>
                                <span class="magicai-chat--list-chat--action" data-action="delete" data-postid="${response.data}">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M4 8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8ZM6 10V20H18V10H6ZM9 12H11V18H9V12ZM13 12H15V18H13V12ZM7 5V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V5H22V7H2V5H7ZM9 4V5H15V4H9Z"></path></svg>
                                    Delete
                                </span>
                            </div>
                        </details>
                    </div>
                </div>
                `);
            }
            message_wrapper.append(`
            <div class="magicai-chat--message ai">
                <span class="text">
                <img width="32" src="${magicai_js_options.logo_url}">    
                How can I help you today?</span>
            </div>
            `);
        } );
        
    });

    $(document).on("click", ".magicai-chat--toggle", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        $this.toggleClass('toggle');
        var message_wrapper = $this.parents().eq(2).find('.magicai-chat--list');
        message_wrapper.toggleClass('toggle');
        
    });

    $(document).on("click", ".magicai-chat--list-chat--trigger", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let message_label = $this.parent().find('.message').html();
        let header_title = $this.parents().eq(2).find('.magicai-chat--header-title');

        var post_id = $this.attr('data-postid');
        var message_wrapper = $this.parents().eq(2).find('.magicai-chat--message-list');
            message_wrapper.empty()
            message_wrapper.attr('data-postid', post_id );

        var list_wrapper = $this.parents().eq(2).find('.magicai-chat--list');
            list_wrapper.removeClass('toggle');

        var mobile_toggle = $this.parents().eq(2).find('.magicai-chat--toggle');
            mobile_toggle.removeClass('toggle');

        header_title.html(message_label);

        jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_get_chat', post_id: post_id }, function ( response ) {
            message_wrapper.html(response.data);
            $('.magicai-chat--message-list').scrollTop($('.magicai-chat--message-list')[0].scrollHeight - $('.magicai-chat--message-list')[0].clientHeight);
        } );
        
    });

    $(document).on("click", ".magicai-chat--list-chat--action", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let parent = $this.parents().eq(3);
        let message_label = parent.find('.message');
        let header_title = $this.parents().eq(5).find('.magicai-chat--header-title');
        let post_id = $this.attr('data-postid');
        let action = $this.attr('data-action');

        if ( action == 'delete' ) {
            jQuery.confirm( {
                columnClass: 'magicai-modal',
                title: 'Delete',
                content: 'Are you sure?',
                closeIcon: true,
                closeIconClass: 'dashicons dashicons-no',
                buttons: {
                    new: {
                        btnClass: 'btn-blue',
                        text: 'Delete',
                        action: function () {
                            jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_delete_chat', post_id: post_id }, function ( response ) {
                                parent.remove();
                                header_title.append( ' (deleted)' );
                                notyf.success( response.data );
                            });
                        }
                    },
                }
            } );
        } else if ( action == 'edit' ) {
            jQuery.confirm( {
                columnClass: 'magicai-modal',
                title: 'Edit Chat Name',
                content: `<input type="text" placeholder="Enter a new chat name" class="name" style="border: 1px solid #ddd;width: 99%;" value="${message_label.html()}" required />`,
                closeIcon: true,
                closeIconClass: 'dashicons dashicons-no',
                buttons: {
                    new: {
                        btnClass: 'btn-blue',
                        text: 'Save',
                        action: function () {
                            var name = this.$content.find('.name').val();
                            if(!name){
                                jQuery.alert('Enter a chat name.');
                                return false;
                            }
                            jQuery.post( magicai_js_options.ajaxurl, { action: 'magicai_edit_chat_name', post_id: post_id, title: name }, function ( response ) {
                                message_label.html( name );
                                header_title.html( name );
                                notyf.success( response.data );
                            });
                           
                        }
                    },
                }
            } );
        }
        
    });

    $(document).on("click", ".magicai-chat--message-action .copy", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let $this = $(this);
        let message = $this.parents().eq(1).find('.text div');
        let divContent = message.html();
        let textElement = document.createElement("textarea");
        textElement.value = divContent;
        document.body.appendChild(textElement);
        textElement.select();
        document.execCommand("copy");
        document.body.removeChild(textElement);
        notyf.success( { message: 'Copied', duration: 1200 });

    });

    $(document).on("click", "#chat-form .upload", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        e.preventDefault();

        // Declare a variable for the media upload window
        var mediaUploader;
        var $this = $(this);

        let chat_type = $this.parent().attr('data-type');

        var types = [];

        if ( chat_type == 'pdf' ){
            types = ['application/pdf'];
        } else {
            types = ['image/png', 'image/jpeg', 'image/webp'];
        }

        // Open the media upload window
        if (mediaUploader) {
            mediaUploader.open();
            return;
        }

        // Create a new media upload window
        mediaUploader = wp.media({
            title: 'Select Media',
            button: {
                text: 'Select Media'
            },
            multiple: false, // Set to 'false' if you want to select a single media item
            library: {
                type: types // Specify your custom MIME types here
            }
        });

        // Action when media is selected
        mediaUploader.on('select', function() {
            var attachment = mediaUploader.state().get('selection').first().toJSON();

            if ( chat_type == 'pdf' ) {
                $this.parent().attr('data-id', attachment.id);
                $this.parent().attr('data-filename', attachment.filename);
                $this.parent().attr('data-url', attachment.url);
            } else {
                fetch(attachment.url)
                .then((res) => res.blob())
                .then((blob) => {
                    // Read the Blob as DataURL using the FileReader API
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        $this.parent().attr('data-image', reader.result);
                        // Logs data:image/jpeg;base64,wL2dvYWwgbW9yZ...
                    };
                    reader.readAsDataURL(blob);
                });
            }

            $this.remove('.badge');
            $this.append('<span class="badge"><span class="text">1</span><span class="close">x</span></span>');
        });

        // Open the media upload window
        mediaUploader.open();
    });

    $(document).on("click", ".magicai-chat--form button.upload .badge", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        $(this).closest('.magicai-chat--form').removeAttr('data-image');
        $(this).closest('.magicai-chat--form').removeAttr('data-id');
        $(this).closest('.magicai-chat--form').removeAttr('data-filename');
        $(this).closest('.magicai-chat--form').removeAttr('data-url');
        $(this).remove();


        e.stopPropagation();

    });

    // Speech to Text
    $(document).on("submit", ".magicai-generator.speech-to-text", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');

        if ( !form.find('#file').val() || form.find('#file').val() == 'undefined' ) {
           notyf.error('Select a file');
           return false;
        }

        var form_data = new FormData();
        form_data.append('file', form.find('#file').val());
        form_data.append('action', 'magicai_transcribe_audio');

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.data);
                }
                if ( response.output ) {
                    notyf.success('Generated!');
                    if ( result.hasClass('default') ){
                        result.empty();
                        result.removeClass('default');
                    }
                    result.prepend(response.output);
                }
			},
		});

        e.preventDefault();

    });

    $(document).on("click", ".magicai-generator.speech-to-text .media-uploader", function (e) {

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
            title: 'Select Media',
            button: {
                text: 'Select Media'
            },
            multiple: false, // Set to 'false' if you want to select a single media item
            library: {
                type: ['audio/mpeg', 'audio/mp4', 'audio/mpeg', 'audio/mpga', 'audio/x-m4a', 'audio/wav', 'video/webm'] // Specify your custom MIME types here
            }
        });

        // Action when media is selected
        mediaUploader.on('select', function() {
            var attachment = mediaUploader.state().get('selection').first().toJSON();
            var mediaUrl = attachment.id;

            // Copy the URL of the selected media to the input field
            $this.next('input').val(mediaUrl);
            $this.find('.file-name').remove();
            $this.append(`<span class="file-name">(${attachment.title})</span>`);
        });

        // Open the media upload window
        mediaUploader.open();
    });

    // YouTube Video to Post
    $(document).on("submit", ".magicai-generator.yt-post-generator", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');

        var form_data = new FormData();
        form_data.append('action', 'magicai_generate_yt_post');
        form_data.append('url', form.find('#url').val());
        form_data.append('language', form.find('#language').val());
        form_data.append('yt_action', form.find('#action').val());
        form_data.append('atts', form.find('#atts').val() ?? '');

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                }
                if ( response.output ) {
                    notyf.success('Generated!');
                    result.empty();
                    result.removeClass('default');
                    result.append(response.output);
                    wp.editor.remove('yt_post_content_1');
                    wp.editor.initialize('yt_post_content_1', {tinymce: true,height: 300});
                    $( '#yt_post_content_1_ifr' ).css( 'height', '300px' );
                }
			},
		});

        e.preventDefault();

    });

    // RSS to Post
    $(document).on("click", ".magicai-generator.rss-post-generator .fetch-rss", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        let form = $(this).closest('form');
        
        if ( !form.find('#url').val() ) {
            // console.log(form.find('#url').val())
            notyf.error('Enter the RSS URL!');
            return false;
        }
        var form_data = new FormData();
        form_data.append('action', 'magicai_fetch_rss');
        form_data.append('url', form.find('#url').val());
        var titleSelect = form.find('#title'); 

        $.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                if ( response.error ) {
                    notyf.error(response.message);
                }
                if ( response.output ) {
                    notyf.success('Generated!');
                    titleSelect.empty();
                    titleSelect.append(response.output);
                   
                }
			},
		});

    });

    $(document).on("submit", ".magicai-generator.rss-post-generator", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');

        var form_data = new FormData();
        form_data.append('action', 'magicai_generate_rss_post');
        form_data.append('title', form.find('#title').val());
        form_data.append('url', form.find('#url').val());
        form_data.append('language', form.find('#language').val());
        form_data.append('maximum_lenght', form.find('#maximum_lenght').val());
        form_data.append('number_of_results', form.find('#number_of_results').val());
        form_data.append('temperature', form.find('#temperature').val());
        form_data.append('tone', form.find('#tone').val());
        form_data.append('atts', form.find('#atts').val() ?? '');

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                }
                if ( response.output ) {
                    notyf.success('Generated!');
                    result.empty();
                    result.removeClass('default');
                    result.append(response.output);
                    for ( let i = 0; i < response.n; i++ ) {
                        wp.editor.remove('post_content_' + i);
                        wp.editor.initialize('post_content_' + i, {tinymce: true,height: 300});
                        $( '#post_content_' +i+ '_ifr' ).css( 'height', '300px' );
                    }
                }
			},
		});

        e.preventDefault();

    });

    // Documents Filter
    $(document).ready(function() {
        $(document).on('click', '.magicai-documents--filter div', function() {
            $('.magicai-documents--filter div').removeClass('selected');
            $(this).addClass('selected');
            $('.magicai-documents--list-item').hide();
            var selectedClass = $(this).attr('data-filter');
            if (selectedClass === 'all') {
                $('.magicai-documents--list-item').show();
            } else {
                $('.magicai-documents--list-item.' + selectedClass).show();
            }
        });
    });

    // ChatBot Template Settings
    $(document).on("submit", ".magicai-chatbot-widget", function (e) {

        let form = $(this);
        let result = $(this).parent().parent().find('.generator-result');
        let button = form.find('.btn');
        var form_data = new FormData();
        e.preventDefault();

        form_data.append('action', 'magicai_chatbot_settings');
        form_data.append('nonce', $('#nonce').val());
        form_data.append('status', $('#status').val());
        form_data.append('template', $('#template').val());
        form_data.append('position', $('#position').val());
        form_data.append('limit', $('#limit').val());
        form_data.append('limit_per_seconds', $('#limit_per_seconds').val());
        form_data.append('is_user_logged_in', $('#is_user_logged_in').is(":checked") ? 1 : 0 );

		$.ajax({
			url: magicai_js_options.ajaxurl,
			type: 'POST',
			contentType: false,
            processData: false,
			data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
                button.attr('disabled', true);
            },
			success: function ( response ) {
                // console.log(response);
                $('body').removeClass('magicai-loading');
                button.removeAttr('disabled');
                if ( response.error ) {
                    notyf.error(response.message);
                    return;
                } else {
                    notyf.success(response.message);
                    setTimeout(function() {
                        location.reload();
                      }, 3000);
                }
				
			}
		});

        e.preventDefault();

    });

    // Fine Tune - add new
    $(document).on("click", ".magicai-add--fine-tune", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }

        var currentModal = jQuery.confirm( {
            columnClass: 'magicai-modal',
            title: magicai_js_options.modal.title.add_fine_tune,
            content: magicai_js_options.modal.content.add_fine_tune,
            closeIcon: true,
            closeIconClass: 'dashicons dashicons-no',
            buttons: {
                add: {
                    btnClass: 'btn-blue',
                    text: '',
                    action: function () {

                        $('.magicai-modal').addClass('is-loading');

                        var form_data = new FormData();
                        form_data.append('action', 'magicai_create_fine_tune');
                        form_data.append('title', this.$content.find('#title').val());
                        form_data.append('model', this.$content.find('#model').val());
                        form_data.append('purpose', this.$content.find('#purpose').val());
                        
                        if ( this.$content.find('#file').val() != 'undefined' ) {
                            form_data.append('file', this.$content.find('#file').prop('files')[0]);
                        }

                        $.ajax({
                            url: magicai_js_options.ajaxurl,
                            type: 'POST',
                            contentType: false,
                            processData: false,
                            data: form_data,
                            beforeSend: function() {
                                $('body').addClass('magicai-loading');
                            },
                            success: function ( response ) {
                                $('.magicai-modal').removeClass('is-loading');
                                $('body').removeClass('magicai-loading');
                                if ( response.error ) {
                                    notyf.error(response.message);
                                }
                                if ( response.output ) {
                                    $('.magicai-fine-tune-table tbody').prepend(response.output);
                                    currentModal.close();
                                    notyf.success( 'Saved!' );
                                }
                            },
                        });

                        return false;
                    }
                },
            }
        } );

    });

    // Fine Tune - delete
    $(document).on("click", ".magicai-delete--fine-tune", function (e) {

        if ( !magicai_js_options.guest_status ) {
            notyf.error('Please Active The MagicAI');
            return false;
        }
        
        let button = $(this);
        let file_id = button.attr( 'data-file' ); 
        let model = button.attr( 'data-model' );
        let row = button.closest('tr');

        row.remove();

        if ( ! file_id ){
            notyf.error('File not found!');
            return false;
        }

        var currentModal = jQuery.confirm( {
            columnClass: 'magicai-modal',
            title: magicai_js_options.modal.title.delete_fine_tune,
            content: magicai_js_options.modal.content.delete_fine_tune,
            closeIcon: true,
            closeIconClass: 'dashicons dashicons-no',
            buttons: {
                delete: {
                    btnClass: 'btn-blue',
                    text: '',
                    action: function () {

                        var form_data = new FormData();
                        form_data.append('action', 'magicai_delete_fine_tune');
                        form_data.append('file_id', file_id);
                        form_data.append('model', model);

                        $.ajax({
                            url: magicai_js_options.ajaxurl,
                            type: 'POST',
                            contentType: false,
                            processData: false,
                            data: form_data,
                            beforeSend: function() {
                                $('.magicai-modal').addClass('is-loading');
                                $('body').addClass('magicai-loading');
                            },
                            success: function ( response ) {
                                $('.magicai-modal').removeClass('is-loading');
                                $('body').removeClass('magicai-loading');
                                if ( response.error ) {
                                    notyf.error(response.message);
                                }
                                if ( response.deleted ) {
                                    row.remove();
                                    notyf.success( response.message );
                                    currentModal.close();
                                }
                            },
                        });

                        return false;
                    }
                },
                cancel:{

                }
            }
        } );

    });

    // Assistant - Actions
    $(document).on("click", ".magicai-modal.assistant .magicai-assistant-prompt", function (e) {

        var prompt = $(this).attr('data-prompt');

        if ( !prompt ) {
            notyf.error('Prompt is undefined!');
            return false;
        }

        if ( prompt == 'settings-assistant' ){
            window.open($(this).attr('data-href'), '_blank');
            return false;
        }

        magicai_create_assistant_request(prompt);

    });

    // Assistant - Custom Prompt
    $(document).on("submit", "#magicai-assistant-form", function (e) {

        e.preventDefault();

        var prompt = $(this).find('.magicai-assistant-form--prompt').val();

        if ( !prompt ) {
            notyf.error( 'Prompt is empty!' );
            return false;
        }

        magicai_create_assistant_request(prompt);

    });

    // Assistant - Create Request
    function magicai_create_assistant_request( prompt ) {

        if ( magicai_assistant_classic_editor ) {
            $('body').addClass('magicai-loader');
            magicai_assistant_modal.close();
            $.post( magicai_js_options.ajaxurl, { action: 'magicai_block_editor', data: { prompt: prompt, content: magicai_assistant_classic_editor } }, function ( response ) {
                if ( response.error ) {
                    notyf.error(response.message);
                } else {
                    magicai_assistant_classic_editor_ed.execCommand('mceInsertContent', 0, response.output);
                    notyf.success( 'Created!' );
                }
                $('body').removeClass('magicai-loader');
            } );

        } else {
            const selectedBlock = wp.data.select('core/editor').getSelectedBlock();
            if (selectedBlock) {
                const blockContent = selectedBlock.attributes.content;
                const blockClientId = selectedBlock.clientId;
    
                $('.magicai-toolbar .dashicons-magicai-logo').addClass('is-loading');
                magicai_assistant_modal.close();
                $.post( magicai_js_options.ajaxurl, { action: 'magicai_block_editor', data: { prompt: prompt, content: blockContent, clientId: blockClientId } }, function ( response ) {
                    if ( response.error ) {
                        notyf.error( response.message );
                    } else {
                        wp.data.dispatch('core/editor').updateBlockAttributes(blockClientId, { content: response.output })
                        notyf.success( 'Created!' );
                    }
                    jQuery('.magicai-toolbar .dashicons-magicai-logo').removeClass('is-loading');
                } );
            } else {
                notyf.error( 'Select a block!' );
            }
        }
        
    }

     // Documents Shortcode Open
     $(document).on("click", ".magicai-sc--documents-open", function (e) {

        var post_id = $(this).attr('data-postid');

        if ( !post_id ) {
            notyf.error('Documents does not exists!');
            return false;
        }

        var form_data = new FormData();
        form_data.append('action', 'magicai_get_document_modal');
        form_data.append('post_id', post_id);

        $.ajax({
            url: magicai_js_options.ajaxurl,
            type: 'POST',
            contentType: false,
            processData: false,
            data: form_data,
            beforeSend: function() {
                $('body').addClass('magicai-loading');
            },
            success: function ( response ) {
                $('body').removeClass('magicai-loading');
                if ( response.error ) {
                    notyf.error(response.message);
                }
                if ( response.post_title || response.post_content ) {
                    jQuery.confirm( {
                        columnClass: 'magicai-modal hide-buttons',
                        title: response.post_title,
                        content: response.post_content,
                        closeIcon: true,
                        closeIconClass: 'dashicons dashicons-no',
                        onContentReady: function () {
                            Prism.highlightAll();
                        }
                    } );
                    
                }
            },
        });

    });

});

function filter_chat_list( e ) {
    "use strict";
    var filterValue = jQuery(e).val().toLowerCase();
    jQuery('.magicai-chat--list-chat, .magicai-chat--list-date').hide();
    
    jQuery('.magicai-chat--list-chat').filter(function() {
        return jQuery(this).text().toLowerCase().includes(filterValue);
    }).show();

    // jQuery('.magicai-chat--list-chat:contains("' + filterValue + '")').show();
    if (!filterValue) {
        jQuery('.magicai-chat--list-date').show();
    }
}