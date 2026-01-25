<div id="tab-post-generator" class="magicai-settings-form-page">
    <div class="magicai-generator-wrapper">

        <div class="generator-form">
            <form action="" class="magicai-generator post-generator magicai-form flex">
                <div class="form-field">
                    <label for="title">
                        <?php magicai_helper()->label_help_tip( 'Write details about the post you want.' ); ?>
                        <?php esc_html_e( 'Title', 'magicai-wp' ); ?>
                    </label>
                    <input type="text" name="title" id="title" placeholder="<?php esc_attr_e( 'Title', 'magicai-wp' ); ?>" required>
                </div>
                
                <div class="form-field">
                    <label for="tag">
                        <?php magicai_helper()->label_help_tip( 'Enter keywords for SEO-focused posts.' ); ?>
                        <?php esc_html_e( 'Tags', 'magicai-wp' ); ?>
                    </label>
                    <input type="text" name="tag" id="tag" placeholder="<?php esc_attr_e( 'tag1,tag2,tag3...' ); ?>">
                </div>
                
                <div class="form-field">
                    <label for="language">
                        <?php magicai_helper()->label_help_tip( 'Choose the language of the content.' ); ?>
                        <?php esc_html_e( 'Language', 'magicai-wp' ); ?>
                    </label>
                    <select name="language" id="language" required>
                        <?php 
                            foreach ( magicai_helper()->get_const_vars('OPENAI_LANGUAGES') as $language_code => $language_name ) {
                                printf( 
                                    '<option value="%1$s" %2$s>%3$s</option>',
                                    esc_attr( $language_code ),
                                    selected( $language_code, esc_attr( magicai_helper()->get_option( 'openai_default_language' ) ) ),
                                    esc_html( $language_name )
                                );
                            }
                        ?>
                    </select>
                </div>
                
                <div class="form-field">
                    <label for="image">
                        <?php magicai_helper()->label_help_tip( 'Create thumbnails for the post to be created.' ); ?>
                        <?php esc_html_e( 'Generate Thumbnail', 'magicai-wp' ); ?>
                    </label>
                    <select name="image" id="image">
                        <option value="" selected><?php esc_html_e( 'No', 'magicai-wp' ); ?></option>
                        <option value="unsplash" <?php if ( ! magicai_helper()->get_option( 'unsplash_api_key' ) ) echo esc_attr('disabled'); ?>><?php esc_html_e( 'Unsplash', 'magicai-wp' ); ?> <?php if ( ! magicai_helper()->get_option( 'unsplash_api_key' ) ) esc_html_e('(Enter the API key!)', 'magicai-wp');?></option>
                    </select>
                </div>

                <div class="form-field half">
                    <label for="maximum_lenght">
                        <?php magicai_helper()->label_help_tip( 'Maximum content length (word).' ); ?>
                        <?php esc_html_e( 'Maximum Lenght', 'magicai-wp' ); ?>
                    </label>
                    <input type="number" name="maximum_lenght" id="maximum_lenght" value="400" required>
                </div>

                <div class="form-field half">
                    <label for="number_of_results">
                        <?php magicai_helper()->label_help_tip( 'Choose the number of posts to be created.' ); ?>
                        <?php esc_html_e( 'Number of Results', 'magicai-wp' ); ?>
                    </label>
                    <input type="number" name="number_of_results" id="number_of_results" value="1" min="1" max="10" required>
                </div>

                <div class="form-field half">
                    <label for="temperature">
                        <?php magicai_helper()->label_help_tip( 'Determine the creativity level: Professional - more creativity; Economic - less creativity.' ); ?>
                        <?php esc_html_e( 'Creativity', 'magicai-wp' ); ?>
                    </label>
                    <select name="temperature" id="temperature" required>
                        <?php 
                            foreach ( magicai_helper()->get_const_vars('OPENAI_TEMPERATURE') as $temperature_value => $temperature_name ) {
                                printf( 
                                    '<option value="%1$s" %2$s>%3$s</option>',
                                    esc_attr( $temperature_value ),
                                    selected( $temperature_value, esc_attr( magicai_helper()->get_option( 'openai_temperature' ) ) ),
                                    esc_html( $temperature_name )
                                );
                            }
                        ?>
                    </select>
                </div>

                <div class="form-field half">
                    <label for="tone">
                        <?php magicai_helper()->label_help_tip( 'Select the tone in which it will be written.' ); ?>    
                        <?php esc_html_e( 'Tone', 'magicai-wp' ); ?>
                    </label>
                    <select name="tone" id="tone" required>
                        <?php 
                            foreach ( magicai_helper()->get_const_vars('OPENAI_TONE') as $tone ) {
                                printf( 
                                    '<option value="%1$s" %2$s>%3$s</option>',
                                    esc_attr( $tone ),
                                    selected( $tone, esc_attr( magicai_helper()->get_option( 'openai_default_tone' ) ) ),
                                    esc_html( $tone )
                                );
                            }
                        ?>
                    </select>
                </div>

                <?php MagicAI_Shortcodes::instance()->form_data($atts ?? array()); ?>

                <div class="form-field">
                    <button class="btn" type="submit"><?php esc_html_e( 'Generate', 'magicai-wp' ) ?></button>
                </div>

            </form>
        </div>

        <div class="generator-result default">
            <?php magicai_helper()->generator_default_template(); ?>
        </div>

    </div>
</div>