<div id="tab-rss-post-generator" class="magicai-settings-form-page">
    <div class="magicai-generator-wrapper">

        <div class="generator-form rss-post">
            <form action="" class="magicai-generator rss-post-generator magicai-form flex">
                <div class="form-field">
                    <label for="url">
                        <?php magicai_helper()->label_help_tip( 'Enter the RSS Feed URL' ); ?>
                        <?php esc_html_e( 'URL', 'magicai-wp' ); ?>
                    </label>
                    <input type="URL" name="url" id="url" placeholder="<?php esc_attr_e( 'https://example.com/feed', 'magicai-wp' ); ?>" required>
                    <div class="fetch-rss">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="vertical-align: text-bottom;"><path d="M12 4C9.25144 4 6.82508 5.38626 5.38443 7.5H8V9.5H2V3.5H4V5.99936C5.82381 3.57166 8.72764 2 12 2C17.5228 2 22 6.47715 22 12H20C20 7.58172 16.4183 4 12 4ZM4 12C4 16.4183 7.58172 20 12 20C14.7486 20 17.1749 18.6137 18.6156 16.5H16V14.5H22V20.5H20V18.0006C18.1762 20.4283 15.2724 22 12 22C6.47715 22 2 17.5228 2 12H4Z"></path></svg>
                        <?php esc_html_e( 'Fetch RSS', 'magicai-wp' ) ?>
                    </div>
                </div>
                
                <div class="form-field">
                    <label for="title">
                        <?php magicai_helper()->label_help_tip( 'Select the fetched title. The post will be created according to this title.' ); ?>
                        <?php esc_html_e( 'Fetched Post Title', 'magicai-wp' ); ?>
                    </label>
                    <select name="title" id="title" required>
                        <option value=""><?php esc_html_e( 'Enter the Feed URL, please!', 'magicai-wp' ); ?></option>
                    </select>
                </div>
                <hr style="width:100%">
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