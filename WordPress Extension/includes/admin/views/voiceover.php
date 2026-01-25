<div class="wrap magicai-page-wrap">
    <span class="magicai-loader"></span>
    <h1><?php echo esc_html( get_admin_page_title() ); ?></h1>
    <p><?php esc_html_e( 'Bring your words to life.', 'magicai-wp' ); ?></p>
        
    <div class="magicai-generator-wrapper">

        <div class="generator-form openai">

            <div class="voiceover--types">
                <input name="type" type="radio" value="openai" id="openai">
                <label for="openai" class="selected">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M20.5624 10.1875C20.8124 9.5 20.8749 8.8125 20.8124 8.125C20.7499 7.4375 20.4999 6.75 20.1874 6.125C19.6249 5.1875 18.8124 4.4375 17.8749 4C16.8749 3.5625 15.8124 3.4375 14.7499 3.6875C14.2499 3.1875 13.6874 2.75 13.0624 2.4375C12.4374 2.125 11.6874 2 10.9999 2C9.9374 2 8.8749 2.3125 7.9999 2.9375C7.1249 3.5625 6.4999 4.4375 6.1874 5.4375C5.4374 5.625 4.8124 5.9375 4.1874 6.3125C3.6249 6.75 3.1874 7.3125 2.8124 7.875C2.24991 8.8125 2.06241 9.875 2.18741 10.9375C2.31241 12 2.7499 13 3.4374 13.8125C3.1874 14.5 3.1249 15.1875 3.1874 15.875C3.2499 16.5625 3.4999 17.25 3.8124 17.875C4.3749 18.8125 5.1874 19.5625 6.1249 20C7.1249 20.4375 8.1874 20.5625 9.2499 20.3125C9.7499 20.8125 10.3124 21.25 10.9374 21.5625C11.5624 21.875 12.3124 22 12.9999 22C14.0624 22 15.1249 21.6875 15.9999 21.0625C16.8749 20.4375 17.4999 19.5625 17.8124 18.5625C18.4999 18.4375 19.1874 18.125 19.7499 17.6875C20.3124 17.25 20.8124 16.75 21.1249 16.125C21.6874 15.1875 21.8749 14.125 21.7499 13.0625C21.6249 12 21.2499 11 20.5624 10.1875ZM13.0624 20.6875C12.0624 20.6875 11.3124 20.375 10.6249 19.8125C10.6249 19.8125 10.6874 19.75 10.7499 19.75L14.7499 17.4375C14.8749 17.375 14.9374 17.3125 14.9999 17.1875C15.0624 17.0625 15.0624 17 15.0624 16.875V11.25L16.7499 12.25V16.875C16.8124 19.0625 15.0624 20.6875 13.0624 20.6875ZM4.9999 17.25C4.5624 16.5 4.3749 15.625 4.5624 14.75C4.5624 14.75 4.6249 14.8125 4.6874 14.8125L8.6874 17.125C8.8124 17.1875 8.8749 17.1875 8.9999 17.1875C9.1249 17.1875 9.2499 17.1875 9.3124 17.125L14.1874 14.3125V16.25L10.1249 18.625C9.2499 19.125 8.2499 19.25 7.3124 19C6.3124 18.75 5.4999 18.125 4.9999 17.25ZM3.9374 8.5625C4.3749 7.8125 5.0624 7.25 5.8749 6.9375V7.0625V11.6875C5.8749 11.8125 5.8749 11.9375 5.9374 12C5.9999 12.125 6.0624 12.1875 6.1874 12.25L11.0624 15.0625L9.3749 16.0625L5.3749 13.75C4.4999 13.25 3.8749 12.4375 3.6249 11.5C3.3749 10.5625 3.4374 9.4375 3.9374 8.5625ZM17.7499 11.75L12.8749 8.9375L14.5624 7.9375L18.5624 10.25C19.1874 10.625 19.6874 11.125 19.9999 11.75C20.3124 12.375 20.4999 13.0625 20.4374 13.8125C20.3749 14.5 20.1249 15.1875 19.6874 15.75C19.2499 16.3125 18.6874 16.75 17.9999 17V12.25C17.9999 12.125 17.9999 12 17.9374 11.9375C17.9374 11.9375 17.8749 11.8125 17.7499 11.75ZM19.4374 9.25C19.4374 9.25 19.3749 9.1875 19.3124 9.1875L15.3124 6.875C15.1874 6.8125 15.1249 6.8125 14.9999 6.8125C14.8749 6.8125 14.7499 6.8125 14.6874 6.875L9.8124 9.6875V7.75L13.8749 5.375C14.4999 5 15.1874 4.875 15.9374 4.875C16.6249 4.875 17.3124 5.125 17.9374 5.5625C18.4999 6 18.9999 6.5625 19.2499 7.1875C19.4999 7.8125 19.5624 8.5625 19.4374 9.25ZM8.9374 12.75L7.2499 11.75V7.0625C7.2499 6.375 7.4374 5.625 7.8124 5.0625C8.1874 4.4375 8.7499 4 9.3749 3.6875C9.9999 3.375 10.7499 3.25 11.4374 3.375C12.1249 3.4375 12.8124 3.75 13.3749 4.1875C13.3749 4.1875 13.3124 4.25 13.2499 4.25L9.2499 6.5625C9.1249 6.625 9.0624 6.6875 8.9999 6.8125C8.9374 6.9375 8.9374 7 8.9374 7.125V12.75ZM9.8124 10.75L11.9999 9.5L14.1874 10.75V13.25L11.9999 14.5L9.8124 13.25V10.75Z"></path></svg>
                    <?php esc_html_e('OpenAI', 'magicai-wp'); ?>
                </label>
                <input name="type" type="radio" value="google" id="google">
                <label for="google">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 11H20.5329C20.5769 11.3847 20.6 11.7792 20.6 12.1837C20.6 14.9184 19.6204 17.2204 17.9224 18.7837C16.4367 20.1551 14.404 20.9592 11.9796 20.9592C8.46933 20.9592 5.43266 18.947 3.9551 16.0123C3.34695 14.8 3 13.4286 3 11.9796C3 10.5306 3.34695 9.1592 3.9551 7.94698C5.43266 5.01226 8.46933 3 11.9796 3C14.4 3 16.4326 3.88983 17.9877 5.33878L16.5255 6.80101C15.3682 5.68153 13.8028 5 12 5C8.13401 5 5 8.13401 5 12C5 15.866 8.13401 19 12 19C15.5265 19 18.1443 16.3923 18.577 13H12V11Z"></path></svg>
                    <?php esc_html_e('Google', 'magicai-wp'); ?>
                </label>
            </div>

            <form action="" class="magicai-generator openai-voiceover magicai-form">

                <div class="form-field">
                    <label for="openai-languages">
                        <?php magicai_helper()->label_help_tip( 'Choose the language of speech.' ); ?>
                        <?php esc_html_e( 'Language', 'magicai-wp' ); ?>
                    </label>
                    <select name="openai-languages" id="openai-languages" required>
                        <?php 
                            foreach ( magicai_helper()->get_const_vars('OPENAI_TTS_LANGUAGES') as $language_code => $language_name ) {
                                printf( 
                                    '<option value="%1$s" %2$s>%3$s</option>',
                                    esc_attr( $language_code ),
                                    selected( $language_code, esc_attr( 'en-US' ) ),
                                    esc_html( $language_name )
                                );
                            }
                        ?>
                    </select>
                </div>
                
                <div class="form-field">
                    <label for="openai-voice">
                        <?php magicai_helper()->label_help_tip( 'Select the speaker (male/female). Each person has a different tone of voice.' ); ?>
                        <?php esc_html_e( 'Voice', 'magicai-wp' ); ?>
                    </label>
                    <select name="openai-voice" id="openai-voice" required>
                        <option value="alloy"><?php esc_html_e('Alloy (Male)', 'magicai-wp'); ?></option>
                        <option value="echo"><?php esc_html_e('Echo (Male)', 'magicai-wp'); ?></option>
                        <option value="fable"><?php esc_html_e('Fable (Male)', 'magicai-wp'); ?></option>
                        <option value="onyx"><?php esc_html_e('Onyx (Male)', 'magicai-wp'); ?></option>
                        <option value="nova"><?php esc_html_e('Nova (Female)', 'magicai-wp'); ?></option>
                        <option value="shimmer"><?php esc_html_e('Shimmer (Female)', 'magicai-wp'); ?></option>
                    </select>
                </div>

                <div class="form-field">
                    <label for="openai-speech">
                        <?php magicai_helper()->label_help_tip( 'Enter the text to be converted to speech. The language will be automatically detected.' ); ?>
                        <?php esc_html_e( 'Speech', 'magicai-wp' ); ?>
                    </label>
                    <textarea name="openai-speech" id="openai-speech" rows="5" required></textarea>
                </div>

                <div class="form-field">
                    <button class="btn" type="submit"><?php esc_html_e( 'Generate', 'magicai-wp' ) ?></button>
                </div>
            </form>

            <form action="" class="magicai-generator voiceover magicai-form">
                <div class="form-field">
                    <label for="languages">
                        <?php magicai_helper()->label_help_tip( 'Choose the language of speech.' ); ?>
                        <?php esc_html_e( 'Language', 'magicai-wp' ); ?>
                    </label>
                    <select name="languages" id="languages" required>
                        <?php 
                            foreach ( magicai_helper()->get_const_vars('GOOGLE_TTS_LANGUAGES') as $language_code => $language_name ) {
                                printf( 
                                    '<option value="%1$s" %2$s>%3$s</option>',
                                    esc_attr( $language_code ),
                                    selected( $language_code, esc_attr( 'en-US' ) ),
                                    esc_html( $language_name )
                                );
                            }
                        ?>
                    </select>
                </div>

                <div class="form-field">
                    <label for="voice">
                        <?php magicai_helper()->label_help_tip( 'Select the speaker (male/female). Each person has a different tone of voice.' ); ?>
                        <?php esc_html_e( 'Voice', 'magicai-wp' ); ?>
                    </label>
                    <select name="voice" id="voice" required>
                        <option value=""><?php esc_html_e('Select a voice', 'magicai-wp'); ?></option>
                    </select>
                </div>

                <div class="form-field">
                    <label for="pace">
                        <?php magicai_helper()->label_help_tip( 'Choose the speech rate.' ); ?>
                        <?php esc_html_e( 'Pace', 'magicai-wp' ); ?>
                    </label>
                    <select id="pace" name="pace" class="form-control form-select">
                        <option value="x-slow"><?php esc_html_e( 'Very Slow', 'magicai-wp' ); ?></option>
                        <option value="slow"><?php esc_html_e( 'Slow', 'magicai-wp' ); ?></option>
                        <option value="medium" selected><?php esc_html_e( 'Medium', 'magicai-wp' ); ?></option>
                        <option value="fast"><?php esc_html_e( 'Fast', 'magicai-wp' ); ?></option>
                        <option value="x-fast"><?php esc_html_e( 'Ultra Fast', 'magicai-wp' ); ?></option>
                    </select>
                </div>

                <div class="form-field">
                    <label for="break">
                        <?php magicai_helper()->label_help_tip( 'Select the pause duration after each speech.' ); ?>
                        <?php esc_html_e( 'Pause', 'magicai-wp' ); ?>
                    </label>
                    <select id="break" name="break" class="form-control form-select">
                        <option value="0"><?php esc_html_e( '0s', 'magicai-wp' ); ?></option>
                        <option value="1" selected><?php esc_html_e( '1s', 'magicai-wp' ); ?></option>
                        <option value="2"><?php esc_html_e( '2s', 'magicai-wp' ); ?></option>
                        <option value="3"><?php esc_html_e( '3s', 'magicai-wp' ); ?></option>
                        <option value="4"><?php esc_html_e( '4s', 'magicai-wp' ); ?></option>
                    </select>
                </div>

                <div class="form-field">
                    <label>
                        <?php magicai_helper()->label_help_tip( 'Click "add new" to add a new text field for speech.' ); ?>
                        <?php esc_html_e( 'Speeches', 'magicai-wp' ); ?>
                    </label>
                    <button type="button" class="btn add-new-text">+ <?php esc_html_e( 'Add new', 'magicai-wp' ); ?></button>
                    <div class="speeches"></div>
                </div>

                <div class="form-field">
                    <button class="btn" type="submit"><?php esc_html_e( 'Generate', 'magicai-wp' ) ?></button>
                </div>

            </form>
        </div>

        <?php $content = magicai_helper()->get_documents_data( 'voiceover' ); ?>

        <div class="generator-result documents<?php if ( ! $content ) esc_attr_e( ' default' ); ?>">
        <?php 
            if ( ! $content ) {
                magicai_helper()->generator_default_template();
            } else {
                echo $content;
            }
        ?>
        </div>

    </div>

</div>