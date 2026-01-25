<?php

// Meta-Box Generator
// How to use: $meta_value = get_post_meta( $post_id, $field_id, true );
// Example: get_post_meta( get_the_ID(), "my_metabox_field", true );

class MagicAIChatBotOptionsMetabox {

    private $screens = array('magicai-chatbot');

    private $fields = array(
        array(
            'label' => 'Bot Name',
            'id' => '_name',
            'type' => 'text',
            'default' => 'MagicAI Bot',
        ),
        array(
            'label' => 'Role',
            'id' => '_role',
            'type' => 'text',
            'default' => 'Theme Support',
        ),
        array(
            'label' => 'First Message',
            'id' => '_first_message',
            'type' => 'text',
            'default' => 'I am AI Assistant. How can I help you?',
        ),
        array(
            'label' => 'Instructions',
            'id' => '_instructions',
            'type' => 'textarea',
            'default' => "You are an AI assistant providing concise, precise responses. Skilled in multi-turn dialogue, you request clarification as needed. If the context lacks the information, reply, \"I'm sorry, but I can't provide a definite answer based on the available context.\"",
        ),
        array(
            'label' => 'Bot Image',
            'id' => '_image',
            'type' => 'media',
            'returnvalue' => 'url',
            'default' => MAGICAI_URL . 'assets/img/logo.svg'
        ),
        array(
            'label' => 'Trigger Background',
            'id' => '_trigger_bg',
            'type' => 'color',
            'default' => '#ffffff'
        ),
        array(
            'label' => 'Chat Background',
            'id' => '_chat_bg',
            'type' => 'color',
            'default' => '#ffffff'
        ),
        array(
            'label' => 'Message Background (AI)',
            'id' => '_message_ai_bg',
            'type' => 'color',
            'default' => '#dddddd'
        ),
        array(
            'label' => 'Message Color (AI)',
            'id' => '_message_ai_color',
            'type' => 'color',
            'default' => '#000000'
        ),
        array(
            'label' => 'Message Background (User)',
            'id' => '_message_bg',
            'type' => 'color',
            'default' => '#ffffff'
        ),
        array(
            'label' => 'Message Color (User)',
            'id' => '_message_color',
            'type' => 'color',
            'default' => '#000000'
        ),
        array(
            'label' => 'Chat Title Color',
            'id' => '_title_color',
            'type' => 'color',
            'default' => '#000000'
        ),
        array(
            'label' => 'Chat Subtitle Color',
            'id' => '_subtitle_color',
            'type' => 'color',
            'default' => '#000000'
        ),
        array(
            'label' => 'Chat Title Border Color',
            'id' => '_title_border_color',
            'type' => 'color',
            'default' => '#dddddd'
        ),
        array(
            'label' => 'Input Background',
            'id' => '_input_bg',
            'type' => 'color',
            'default' => '#ffffff'
        ),
        array(
            'label' => 'Input Color',
            'id' => '_input_color',
            'type' => 'color',
            'default' => '#000000'
        ),
        array(
            'label' => 'Input Border Color',
            'id' => '_input_border_color',
            'type' => 'color',
            'default' => '#dddddd'
        ),
        array(
            'label' => 'Submit Button Color',
            'id' => '_btn_color',
            'type' => 'color',
            'default' => '#000000'
        ),
        array(
            'label' => 'Width',
            'id' => '_width',
            'type' => 'text',
            'default' => '360px',
        ),
        array(
            'label' => 'Height',
            'id' => '_height',
            'type' => 'text',
            'default' => '420px',
        )  
    );

    public function __construct() {
        add_action( 'add_meta_boxes', array( $this, 'add_meta_boxes' ) );
        add_action( 'save_post', array( $this, 'save_fields' ) );
        add_action( 'admin_footer', array( $this, 'add_media_fields' ) );
    }

    public function add_meta_boxes() {
        foreach ( $this->screens as $s ) {
            add_meta_box(
                'MagicAIChatBotOptions',
                esc_html__( 'MagicAI ChatBot Options', 'magicai-wp' ),
                array( $this, 'meta_box_callback' ),
                $s,
                'normal',
                'high'
            );
            add_meta_box(
                'MagicAIChatBotTraining',
                esc_html__( 'MagicAI ChatBot Training', 'magicai-wp' ),
                array( $this, 'training_meta_box_callback' ),
                $s,
                'normal',
                'high'
            );
            add_meta_box(
                'MagicAIChatBotCoversations',
                esc_html__( 'MagicAI ChatBot Conversations', 'magicai-wp' ),
                array( $this, 'conversations_meta_box_callback' ),
                $s,
                'normal',
                'high'
            );
        }
    }

    public function training_meta_box_callback() {
        ?>
            <div class="magicai-chatbot--train" style="margin-top: 20px">
            
                <div class="magicai-chatbot--train-nav magicai-custom-tab--nav">
                    <div data-href="website" class="magicai-custom-tab--nav-item active">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M15.3873 13.4975L17.9403 20.5117L13.2418 22.2218L10.6889 15.2076L6.79004 17.6529L8.4086 1.63318L19.9457 12.8646L15.3873 13.4975ZM15.3768 19.3163L12.6618 11.8568L15.6212 11.4459L9.98201 5.9561L9.19088 13.7863L11.7221 12.1988L14.4371 19.6583L15.3768 19.3163Z"></path></svg>
                        Website
                    </div>
                    <div data-href="pdf" class="magicai-custom-tab--nav-item">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M5 4H15V8H19V20H5V4ZM3.9985 2C3.44749 2 3 2.44405 3 2.9918V21.0082C3 21.5447 3.44476 22 3.9934 22H20.0066C20.5551 22 21 21.5489 21 20.9925L20.9997 7L16 2H3.9985ZM10.4999 7.5C10.4999 9.07749 10.0442 10.9373 9.27493 12.6534C8.50287 14.3757 7.46143 15.8502 6.37524 16.7191L7.55464 18.3321C10.4821 16.3804 13.7233 15.0421 16.8585 15.49L17.3162 13.5513C14.6435 12.6604 12.4999 9.98994 12.4999 7.5H10.4999ZM11.0999 13.4716C11.3673 12.8752 11.6042 12.2563 11.8037 11.6285C12.2753 12.3531 12.8553 13.0182 13.5101 13.5953C12.5283 13.7711 11.5665 14.0596 10.6352 14.4276C10.7999 14.1143 10.9551 13.7948 11.0999 13.4716Z"></path></svg>
                        PDF
                    </div>
                    <div data-href="text" class="magicai-custom-tab--nav-item">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M13 6V21H11V6H5V4H19V6H13Z"></path></svg>
                        Text
                    </div>
                    <div data-href="qa" class="magicai-custom-tab--nav-item">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M5.76282 17H20V5H4V18.3851L5.76282 17ZM6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455ZM11 14H13V16H11V14ZM8.56731 8.81346C8.88637 7.20919 10.302 6 12 6C13.933 6 15.5 7.567 15.5 9.5C15.5 11.433 13.933 13 12 13H11V11H12C12.8284 11 13.5 10.3284 13.5 9.5C13.5 8.67157 12.8284 8 12 8C11.2723 8 10.6656 8.51823 10.5288 9.20577L8.56731 8.81346Z"></path></svg>
                        Q&A
                    </div>
                </div>
                <div class="magicai-chatbot--train-url active magicai-custom-tab--content" data-content="website">
                    <div class="magicai-form">
                        <div class="form-field relative">
                            <label for="url">
                                <?php magicai_helper()->label_help_tip( 'Enter the website URL. Example: https://magicaidocs-wp.liquid-themes.com' ); ?>
                                <?php esc_html_e( 'Crawl a website', 'magicai-wp' ); ?>
                            </label>
                            <input type="URL" name="url" id="url" value="https://magicaidocs-wp.liquid-themes.com/" placeholder="<?php esc_attr_e( 'https://magicaidocs-wp.liquid-themes.com', 'magicai-wp' ); ?>" required>
                            <div class="btn-fetch" data-type="chatbot_train_url">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="vertical-align: text-bottom;"><path d="M12 4C9.25144 4 6.82508 5.38626 5.38443 7.5H8V9.5H2V3.5H4V5.99936C5.82381 3.57166 8.72764 2 12 2C17.5228 2 22 6.47715 22 12H20C20 7.58172 16.4183 4 12 4ZM4 12C4 16.4183 7.58172 20 12 20C14.7486 20 17.1749 18.6137 18.6156 16.5H16V14.5H22V20.5H20V18.0006C18.1762 20.4283 15.2724 22 12 22C6.47715 22 2 17.5228 2 12H4Z"></path></svg>
                                <?php esc_html_e( 'Fetch Data', 'magicai-wp' ) ?>
                            </div>
                        </div>
                        <div class="form-field relative">
                            <label for="url">
                                <?php magicai_helper()->label_help_tip( 'Enter the single URL. Example: https://magicaidocs-wp.liquid-themes.com/installation' ); ?>
                                <?php esc_html_e( 'Crawl a single URL', 'magicai-wp' ); ?>
                            </label>
                            <input type="URL" name="url" id="url" value="https://magicaidocs-wp.liquid-themes.com/installation" placeholder="<?php esc_attr_e( 'https://magicaidocs-wp.liquid-themes.com/installation', 'magicai-wp' ); ?>" required>
                            <div class="btn-fetch" data-type="chatbot_train_single_url">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="vertical-align: text-bottom;"><path d="M12 4C9.25144 4 6.82508 5.38626 5.38443 7.5H8V9.5H2V3.5H4V5.99936C5.82381 3.57166 8.72764 2 12 2C17.5228 2 22 6.47715 22 12H20C20 7.58172 16.4183 4 12 4ZM4 12C4 16.4183 7.58172 20 12 20C14.7486 20 17.1749 18.6137 18.6156 16.5H16V14.5H22V20.5H20V18.0006C18.1762 20.4283 15.2724 22 12 22C6.47715 22 2 17.5228 2 12H4Z"></path></svg>
                                <?php esc_html_e( 'Fetch Data', 'magicai-wp' ) ?>
                            </div>
                        </div>
                    </div>
                    <?php 
                        if ( $_magicai_train_url = get_post_meta( get_the_ID(), '_magicai_train_url', true ) ) {
                            echo MagicAI_ChatBot::instance()->crawl_result_table( $_magicai_train_url, get_post_meta( get_the_ID(), '_magicai_trained_url', true ) ?? array() );
                        }
                    ?>
                </div>

                <div class="magicai-chatbot--train-pdf magicai-custom-tab--content" data-content="pdf">
                    <button type="button" class="magicai-btn magicai-chatbot--train-pdf-file"><?php esc_html_e( 'Add PDF and Train', 'magicai-wp' ); ?></button>
                    <?php 
                        if ( $magicai_trained_pdf = get_post_meta( get_the_ID(), '_magicai_trained_pdf', true ) ) {
                            echo MagicAI_ChatBot::instance()->pdf_result_table( $magicai_trained_pdf ?? array() );
                        }
                    ?>
                </div>
                <div class="magicai-chatbot--train-text magicai-custom-tab--content" data-content="text">
                    <button type="button" class="magicai-btn magicai-chatbot--train-text-btn"><?php esc_html_e( 'Add Text and Train', 'magicai-wp' ); ?></button>
                    <?php 
                        if ( $magicai_trained_text = get_post_meta( get_the_ID(), '_magicai_trained_text', true ) ) {
                            echo MagicAI_ChatBot::instance()->text_result_table( $magicai_trained_text ?? array() );
                        }
                    ?>
                </div>
                <div class="magicai-chatbot--train-qa magicai-custom-tab--content" data-content="qa">
                    <button type="button" class="magicai-btn magicai-chatbot--train-qa-btn"><?php esc_html_e( 'Add Q&A and Train', 'magicai-wp' ); ?></button>
                    <?php 
                        if ( $magicai_trained_qa = get_post_meta( get_the_ID(), '_magicai_trained_qa', true ) ) {
                            echo MagicAI_ChatBot::instance()->qa_result_table( $magicai_trained_qa ?? array() );
                        }
                    ?>
                </div>

            </div>
        <?php
    }

    public function conversations_meta_box_callback() {

        $chats = get_posts( [
            'post_type' => "magicai-chat",
            'posts_per_page' => -1,
            'meta_query' => [
                [
                    'key' => '_magicai_chatbot_template_id',
                    'value' => get_the_ID(),
                    'compare' => '=',
                ]
            ]
        ] );

        $conversation = count( $chats );
        $message_count_total = 0;

        foreach ( $chats as $chat ) {
            $bot_image = get_post_meta( get_the_ID(), '_image', true );
            $bot_name = get_post_meta( get_the_ID(), '_name', true );
            $bot_role = get_post_meta( get_the_ID(), '_role', true );
            $user_ip = get_post_meta( $chat->ID, '_magicai_user_ip', true );
            $data = get_post_meta( $chat->ID, '_magicai_messages', true );
            $user_id = get_post_meta( $chat->ID, '_magicai_user_id', true );

            $user_data = get_userdata( $user_id );
            $user_name = $user_data->display_name ? '<span class="dashicons dashicons-admin-users"></span> ' . $user_data->display_name : '';

            $html .= sprintf(
                '<div class="magicai-chatbot-widget--message-list--item" data-id="%5$s">
                    <img src="%1$s" width="28" height="28" class="%2$s" alt="ChatBot">
                    <div class="magicai-chatbot-widget--message-list--item-title">
                        <span class="meta">%6$s %7$s</span>
                        %3$s
                        <span>%4$s</span>
                    </div>
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M13.1717 12.0007L8.22192 7.05093L9.63614 5.63672L16.0001 12.0007L9.63614 18.3646L8.22192 16.9504L13.1717 12.0007Z"></path></svg>
                </div>',
                esc_attr( $bot_image ),
                esc_attr( strpos( basename( $bot_image ), '.svg' ) !== false ? 'svg' : '' ),
                esc_html( wp_trim_words( end($data)['content'], 6 ) ),
                esc_html( $bot_name . ' &bull; ' . $bot_role ),
                esc_attr( $chat->ID ),
                '<span class="dashicons dashicons-admin-site"></span>',
                "$user_ip $user_name"

            );
        }

        ?>

        
        <div id="magicai-chatbot--conversations" class="<?php echo esc_attr( !$conversation ? 'no-conversation' : 'init' ); ?>">

            <?php if ( $conversation ) { ?>
                <div class="magicai-chatbot--conversations--message-list">
                    <?php echo $html; ?>
                
                </div>
                <div class="magicai-chatbot--conversations--message-area">
                    
                    <div style="display:grid;grid-template-columns:180px 180px 1fr;gap:24px">
                        <div class="loader">
                            <div class="bar"></div>
                            <div class="bar" style="width:100%"></div>
                            <div class="bar" style="width:40%"></div>
                        </div>
                        <div class="loader">
                            <div class="bar"></div>
                            <div class="bar" style="width:100%"></div>
                            <div class="bar" style="width:40%"></div>
                        </div>
                        <div class="loader" style="justify-content:center;align-items:end;">
                            <div class="bar" style="width:100px;height:40px;"></div>
                        </div>
                    </div>
                    
                    <div class="magicai-chatbot--conversations-message-actions"></div>
                    <div class="loader">
                        <div class="bar"></div>
                        <div class="bar" style="width:100%"></div>
                        <div class="bar" style="width:40%"></div>
                    </div>
                    <div class="magicai-chatbot-widget--messages">
                        <?php magicai_helper()->generator_default_template( esc_html__( 'Select a conversation from the left menu to see the details.', 'magicai-wp' ) ); ?>
                    </div>
                </div>
            <?php 
                } else { 
                    magicai_helper()->generator_default_template( esc_html__( 'There is no conversations for this ChatBot yet.', 'magicai-wp' ) );
                }
            ?>

        </div>
        <?php
    }

    public function meta_box_callback( $post ) {
        wp_nonce_field( 'MagicAIChatBotOptions_data', 'MagicAIChatBotOptions_nonce' ); 
        $this->field_generator( $post );
    }

    public function field_generator( $post ) {
        $output = '';
        foreach ( $this->fields as $field ) {
        $label = '<label for="' . $field['id'] . '">' . $field['label'] . '</label>';
        $meta_value = get_post_meta( $post->ID, $field['id'], true );
        if ( empty( $meta_value ) ) {
            if ( isset( $field['default'] ) ) {
            $meta_value = $field['default'];
            }
        }
        switch ( $field['type'] ) {
            case 'textarea':
            $input = sprintf(
                '<textarea style="width: 100%%" id="%s" name="%s" rows="5">%s</textarea>',
                $field['id'],
                $field['id'],
                $meta_value
            );
            break;
    
            case 'media':
            $meta_url = '';
            if ($meta_value) {
                if ($field['returnvalue'] == 'url') {
                $meta_url = $meta_value;
                } else {
                $meta_url = wp_get_attachment_url($meta_value);
                }
            }
            $input = sprintf(
                '<input style="display:none;" id="%s" name="%s" type="text" value="%s" data-return="%s"><div id="preview%s" style="margin-bottom:12px;background-color:#fafafa;margin-right:12px;border:1px solid #eee;width: 150px;height:150px;background-image:url(%s);background-size:cover;background-repeat:no-repeat;background-position:center;"></div><input style="width: 15%%!important;margin-right:5px;" class="button new-media" id="%s_button" name="%s_button" type="button" value="Select" /><input style="width: 15%%!important;" class="button remove-media" id="%s_buttonremove" name="%s_buttonremove" type="button" value="Delete" />',
                $field['id'],
                $field['id'],
                $meta_value,
                $field['returnvalue'],
                $field['id'],
                $meta_url,
                $field['id'],
                $field['id'],
                $field['id'],
                $field['id']
            );
            break;
    
            default:
            $input = sprintf(
            '<input %s id="%s" name="%s" type="%s" value="%s">',
            $field['type'] !== 'color' ? 'style="width: 100%"' : 'style="width: 100px!important"',
            $field['id'],
            $field['id'],
            $field['type'],
            $meta_value
            );
        }
        $output .= $this->format_rows( $label, $input );
        }
        echo '<div class="magicai-form">' . $output . '</div>';
    }

    public function format_rows( $label, $input ) {
        return '<div class="form-field" style="margin-top: 10px;"><strong>'.$label.'</strong><div>'.$input.'</div></div>';
    }

    
    public function add_media_fields() {
    ?>
    <script>
        jQuery(document).ready(function($){
        if ( typeof wp.media !== 'undefined' ) {
            var _custom_media = true,
            _orig_send_attachment = wp.media.editor.send.attachment;
            $(document).on('click', '.new-media', function(e) {
            var send_attachment_bkp = wp.media.editor.send.attachment;
            var button = $(this);
            var id = button.attr('id').replace('_button', '');
            _custom_media = true;
            wp.media.editor.send.attachment = function(props, attachment){
                if ( _custom_media ) {
                if ($('input#' + id).data('return') == 'url') {
                    $('input#' + id).val(attachment.url);
                } else {
                    $('input#' + id).val(attachment.id);
                }
                $('div#preview'+id).css('background-image', 'url('+attachment.url+')');
                } else {
                return _orig_send_attachment.apply( this, [props, attachment] );
                };
            }
            wp.media.editor.open(button);
            return false;
            });
            $('.add_media').on('click', function(){
            _custom_media = false;
            });
            $('.remove-media').on('click', function(){
            var parent = $(this).parent();
            parent.find('input[type="text"]').val('');
            parent.find('div').css('background-image', 'url()');
            });
        }
        });
        
    </script><?php
    }


    public function save_fields( $post_id ) {
        if ( !isset( $_POST['MagicAIChatBotOptions_nonce'] ) ) {
            return $post_id;
        }
        $nonce = $_POST['MagicAIChatBotOptions_nonce'];
        if ( !wp_verify_nonce( $nonce, 'MagicAIChatBotOptions_data' ) ) {
            return $post_id;
        }
        if ( defined( 'DOING_AUTOSAVE' ) && DOING_AUTOSAVE ) {
            return $post_id;
        }
        foreach ( $this->fields as $field ) {
            if ( isset( $_POST[ $field['id'] ] ) ) {
                switch ( $field['type'] ) {
                case 'email':
                    $_POST[ $field['id'] ] = sanitize_email( $_POST[ $field['id'] ] );
                    break;
                case 'text':
                    $_POST[ $field['id'] ] = sanitize_text_field( $_POST[ $field['id'] ] );
                    break;
                }
                update_post_meta( $post_id, $field['id'], $_POST[ $field['id'] ] );
            } else if ( $field['type'] === 'checkbox' ) {
                update_post_meta( $post_id, $field['id'], '0' );
            }
        }
    }

}

if (class_exists('MagicAIChatBotOptionsMetabox')) {
    new MagicAIChatBotOptionsMetabox;
};

      