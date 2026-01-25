<div id="tab-yt-post-generator" class="magicai-settings-form-page">
    <div class="magicai-generator-wrapper">

        <div class="generator-form">
            <form action="" class="magicai-generator yt-post-generator magicai-form">
                
                <div class="form-field">
                    <label for="url">
                        <?php magicai_helper()->label_help_tip( 'Enter the YouTube video URL.' ); ?>
                        <?php esc_html_e( 'YouTube Video URL', 'magicai-wp' ); ?>
                    </label>
                    <input type="url" name="url" id="url" required placeholder="https://www.youtube.com/watch?v=--khbXchTeE">
                </div>

                <div class="form-field">
                    <label for="action">
                        <?php magicai_helper()->label_help_tip( 'Choose the action in which should be written.' ); ?>
                        <?php esc_html_e( 'Action', 'magicai-wp' ); ?>
                    </label>
                    <select name="action" id="action" required>
                        <option value="blog"><?php esc_html_e('Prepare a Blog Post', 'magicai-wp'); ?></option>
                        <option value="short"><?php esc_html_e('Explain the Main Idea', 'magicai-wp'); ?></option>
                        <option value="list"><?php esc_html_e('Create a List', 'magicai-wp'); ?></option>
                        <option value="tldr"><?php esc_html_e('Create TLDR', 'magicai-wp'); ?></option>
                        <option value="pros_cons"><?php esc_html_e('Prepare Pros and Cons', 'magicai-wp'); ?></option>
                    </select>
                </div>

                <div class="form-field">
                    <label for="language">
                        <?php magicai_helper()->label_help_tip( 'Choose the language in which the post should be written.' ); ?>
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