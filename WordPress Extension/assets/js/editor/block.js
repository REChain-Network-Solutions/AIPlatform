"use strict"; 
var magicai_assistant_modal;
var magicai_assistant_classic_editor;
var magicai_assistant_classic_editor_ed;
const MagicAI_Block_Extended = function ( BlockEdit ) {
	return function ( props ) {
		if ( props.name !== 'core/paragraph' && props.name !== 'core/heading' ) {
			return BlockEdit( props );
		}
		
		return [
			wp.element.createElement(
				wp.blockEditor.BlockControls,
				null,
				wp.element.createElement(
					wp.components.ToolbarButton,
					{
						icon: 'magicai-logo',
						label: 'MagicAI',
						className: 'magicai-toolbar',
						onClick: function () {
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
					}
				)
			),
			BlockEdit( props )
		];
		
	};

}
wp.hooks.addFilter( 'editor.BlockEdit', 'magicai-wp', MagicAI_Block_Extended );
