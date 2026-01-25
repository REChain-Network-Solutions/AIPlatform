"use strict";
var magicai_assistant_modal;
var magicai_assistant_classic_editor;
var magicai_assistant_classic_editor_ed;
// https://developer.wordpress.org/reference/hooks/mce_external_plugins/#user-contributed-notes
(function() {

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

    /* Register the buttons */
    tinymce.create('tinymce.plugins.MagicaiClassicButtons', {
        init : function(ed, url) {
            /**
             * Inserts shortcode content
             */
            ed.addButton( 'magicai_button', {
                title : 'MagicAI',
                classes: 'magicai-classic-toolbar',
                image : magicai_js_options.logo_url,
                onclick : function() {
                    magicai_assistant_classic_editor = ed.selection.getContent();
                    magicai_assistant_classic_editor_ed = ed;
                    if ( magicai_assistant_classic_editor === '') {
                        notyf.error('select a text first!');
                        return false;
                    }
                    magicai_assistant_modal = jQuery.confirm( {
                        columnClass: 'magicai-modal assistant',
                        //type: 'dark',
                        title: magicai_js_options.modal.title.assistant,
                        content: magicai_js_options.modal.content.assistant,
                        closeIcon: true,
                        closeIconClass: 'dashicons dashicons-no',
                        buttons: {},
                } );
                }
            });         

        },
        createControl : function(n, cm) {
            return null;
        },
    });
    /* Start the buttons */
    tinymce.PluginManager.add( 'magicai_classic_button_script', tinymce.plugins.MagicaiClassicButtons );
})();