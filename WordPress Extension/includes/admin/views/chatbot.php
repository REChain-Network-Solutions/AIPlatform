<div class="wrap magicai-page-wrap">
    <span class="magicai-loader"></span>
    <h1><?php echo esc_html( get_admin_page_title() ); ?></h1>
    <p><?php esc_html_e( 'Embed on your site or dashboard easily and interact with your customers.', 'magicai-wp' ) ?></p>
    
    <?php $magicai_chatbot_settings = get_option('magicai_chatbot_settings', array() ); ?>
        
    <div class="magicai-generator-wrapper">

        <div class="generator-form">
            <form class="magicai-chatbot-widget magicai-form">
               
                <div class="form-field">
                    <label for="status">
                        <?php magicai_helper()->label_help_tip( 'You can Show ChatBot Ballon as you like.' ); ?>
                        <?php esc_html_e( 'Show ChatBot Ballon on', 'magicai-wp' ); ?>
                    </label>
                    <select name="status" id="status" required>
                        <option <?php selected( $magicai_chatbot_settings['status'], 'disabled' ); ?> value="disabled"><?php esc_html_e('Disabled', 'magicai-wp'); ?></option>
                        <option <?php selected( $magicai_chatbot_settings['status'], 'frontend' ); ?> value="frontend"><?php esc_html_e('Frontend', 'magicai-wp'); ?></option>
                        <option <?php selected( $magicai_chatbot_settings['status'], 'wp' ); ?> value="wp"><?php esc_html_e('WP Dashboard', 'magicai-wp'); ?></option>
                        <option <?php selected( $magicai_chatbot_settings['status'], 'both' ); ?> value="both"><?php esc_html_e('Both', 'magicai-wp'); ?></option>
                    </select>
                </div>

                <div class="form-field">
                    <label for="template">
                        <?php magicai_helper()->label_help_tip( 'Select a chatbot template.' ); ?>
                        <?php esc_html_e( 'Template', 'magicai-wp' ); ?>
                    </label>
                    <select name="template" id="template">
                        <option value=""><?php esc_html_e('Select a template!', 'magicai-wp'); ?></option>
                        <?php
                             $posts = get_posts(  
                                [
                                    'post_type' => 'magicai-chatbot',
                                    'posts_per_page' => -1,
                                    'post_status' => 'any',
                                ]
                            );         
                            
                            if ( $posts ) {
                                foreach( $posts as $post ) {
                                    printf( 
                                        '<option value="%s" %s>%s</option>',
                                        esc_attr( $post->ID ),
                                        selected( $magicai_chatbot_settings['template'], $post->ID ),
                                        esc_html( $post->post_title )
                                    );
                                }
                            }
                        ?>
                    </select>
                </div>
               
                <div class="form-field">
                    <label for="position">
                        <?php magicai_helper()->label_help_tip( 'Set the ChatBot Ballon position.' ); ?>
                        <?php esc_html_e( 'Position', 'magicai-wp' ); ?>
                    </label>
                    <select name="position" id="position" required>
                        <option <?php selected( $magicai_chatbot_settings['position'], 'bottom-right' ); ?> value="bottom-right"><?php esc_html_e('Bottom Right', 'magicai-wp'); ?></option>
                        <option <?php selected( $magicai_chatbot_settings['position'], 'bottom-left' ); ?> value="bottom-left"><?php esc_html_e('Bottom Left', 'magicai-wp'); ?></option>
                        <option <?php selected( $magicai_chatbot_settings['position'], 'top-right' ); ?> value="top-right"><?php esc_html_e('Top Right', 'magicai-wp'); ?></option>
                        <option <?php selected( $magicai_chatbot_settings['position'], 'top-left' ); ?> value="top-left"><?php esc_html_e('Top Left', 'magicai-wp'); ?></option>
                    </select>
                </div>
               
                <div class="form-field">
                    <label for="limit">
                        <?php magicai_helper()->label_help_tip( 'You can define limitation of messages per 12 hours. Define 0 if you need to disable limitation.' ); ?>
                        <?php esc_html_e( 'Message limit', 'magicai-wp' ); ?>
                    </label>
                    <input type="number" name="limit" id="limit" value="<?php echo esc_attr( $magicai_chatbot_settings['limit'] ?? '0' ); ?>" min="0">
                </div>
               
                <div class="form-field">
                    <label for="limit">
                        <?php magicai_helper()->label_help_tip( 'Define seconds per message limit. Example set 3600 for 1 hour.' ); ?>
                        <?php esc_html_e( 'Limit per seconds', 'magicai-wp' ); ?>
                    </label>
                    <input type="number" name="limit_per_seconds" id="limit_per_seconds" value="<?php echo esc_attr( $magicai_chatbot_settings['limit_per_seconds'] ?? '300' ); ?>" min="0">
                </div>
               
                <div class="form-filed magicai-switch">
                    <?php magicai_helper()->label_help_tip( 'Ensures that the ChatBot Ballon does not appear if the current visitor is not logged in.' ); ?>
                    <span><?php esc_html_e( 'Disable if user not logged in?', 'magicai-wp' ); ?></span>
                    <input type="checkbox" id="is_user_logged_in" name="is_user_logged_in" <?php checked( $magicai_chatbot_settings['is_user_logged_in'], 1 ); ?>/>
                    <label for="is_user_logged_in"></label>
                </div>

                <?php wp_nonce_field( 'magicai_chatbot_settings', 'nonce' ); ?>

                <div class="form-field">
                    <button class="btn" type="submit"><?php esc_html_e( 'Save', 'magicai-wp' ) ?></button>
                </div>

            </form>
        </div>

        <div class="div">
            <div class="magicai-chatbot--create-template">
                <a class="magicai-btn" type="button" href="<?php echo esc_url(add_query_arg(array('post_type'=>'magicai-chatbot'),admin_url('post-new.php'))); ?>" target="_blank">
                    <?php esc_html_e( 'Create ChatBot Template', 'magicai-wp' ); ?>
                </a>
            </div>
            <div class="magicai-chatbot--list">
                <?php 
                    if ( $content = magicai_helper()->get_chatbot_templates( ) ) {
                        echo $content;
                    } else {
                        esc_html_e( 'There is no template. Create template first!', 'magicai-wp' );
                    }
                ?>
            </div>
        </div>

    </div>

</div>