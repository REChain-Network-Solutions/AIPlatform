<div id="tab-image-generator" class="magicai-settings-form-page">
    <div class="magicai-generator-wrapper image">

        <div class="generator-form">
            <form action="" class="magicai-generator image-generator magicai-form <?php if ( isset($atts['hide']) && $atts['hide'] == 'dalle') { echo esc_attr('sd'); } ?>">

                <div class="image-generator--types" <?php if ( isset($atts['hide']) && ($atts['hide'] == 'dalle' || $atts['hide'] == 'sd' ) ) { echo esc_attr('style=display:none;'); } ?>>
                    <input name="type" type="radio" value="dalle" id="dalle" <?php if ( (isset($atts['hide']) && $atts['hide'] == 'sd') || !isset($atts['hide']) ) { echo 'checked'; } ?>>
                    <label for="dalle" class="selected">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M20.5624 10.1875C20.8124 9.5 20.8749 8.8125 20.8124 8.125C20.7499 7.4375 20.4999 6.75 20.1874 6.125C19.6249 5.1875 18.8124 4.4375 17.8749 4C16.8749 3.5625 15.8124 3.4375 14.7499 3.6875C14.2499 3.1875 13.6874 2.75 13.0624 2.4375C12.4374 2.125 11.6874 2 10.9999 2C9.9374 2 8.8749 2.3125 7.9999 2.9375C7.1249 3.5625 6.4999 4.4375 6.1874 5.4375C5.4374 5.625 4.8124 5.9375 4.1874 6.3125C3.6249 6.75 3.1874 7.3125 2.8124 7.875C2.24991 8.8125 2.06241 9.875 2.18741 10.9375C2.31241 12 2.7499 13 3.4374 13.8125C3.1874 14.5 3.1249 15.1875 3.1874 15.875C3.2499 16.5625 3.4999 17.25 3.8124 17.875C4.3749 18.8125 5.1874 19.5625 6.1249 20C7.1249 20.4375 8.1874 20.5625 9.2499 20.3125C9.7499 20.8125 10.3124 21.25 10.9374 21.5625C11.5624 21.875 12.3124 22 12.9999 22C14.0624 22 15.1249 21.6875 15.9999 21.0625C16.8749 20.4375 17.4999 19.5625 17.8124 18.5625C18.4999 18.4375 19.1874 18.125 19.7499 17.6875C20.3124 17.25 20.8124 16.75 21.1249 16.125C21.6874 15.1875 21.8749 14.125 21.7499 13.0625C21.6249 12 21.2499 11 20.5624 10.1875ZM13.0624 20.6875C12.0624 20.6875 11.3124 20.375 10.6249 19.8125C10.6249 19.8125 10.6874 19.75 10.7499 19.75L14.7499 17.4375C14.8749 17.375 14.9374 17.3125 14.9999 17.1875C15.0624 17.0625 15.0624 17 15.0624 16.875V11.25L16.7499 12.25V16.875C16.8124 19.0625 15.0624 20.6875 13.0624 20.6875ZM4.9999 17.25C4.5624 16.5 4.3749 15.625 4.5624 14.75C4.5624 14.75 4.6249 14.8125 4.6874 14.8125L8.6874 17.125C8.8124 17.1875 8.8749 17.1875 8.9999 17.1875C9.1249 17.1875 9.2499 17.1875 9.3124 17.125L14.1874 14.3125V16.25L10.1249 18.625C9.2499 19.125 8.2499 19.25 7.3124 19C6.3124 18.75 5.4999 18.125 4.9999 17.25ZM3.9374 8.5625C4.3749 7.8125 5.0624 7.25 5.8749 6.9375V7.0625V11.6875C5.8749 11.8125 5.8749 11.9375 5.9374 12C5.9999 12.125 6.0624 12.1875 6.1874 12.25L11.0624 15.0625L9.3749 16.0625L5.3749 13.75C4.4999 13.25 3.8749 12.4375 3.6249 11.5C3.3749 10.5625 3.4374 9.4375 3.9374 8.5625ZM17.7499 11.75L12.8749 8.9375L14.5624 7.9375L18.5624 10.25C19.1874 10.625 19.6874 11.125 19.9999 11.75C20.3124 12.375 20.4999 13.0625 20.4374 13.8125C20.3749 14.5 20.1249 15.1875 19.6874 15.75C19.2499 16.3125 18.6874 16.75 17.9999 17V12.25C17.9999 12.125 17.9999 12 17.9374 11.9375C17.9374 11.9375 17.8749 11.8125 17.7499 11.75ZM19.4374 9.25C19.4374 9.25 19.3749 9.1875 19.3124 9.1875L15.3124 6.875C15.1874 6.8125 15.1249 6.8125 14.9999 6.8125C14.8749 6.8125 14.7499 6.8125 14.6874 6.875L9.8124 9.6875V7.75L13.8749 5.375C14.4999 5 15.1874 4.875 15.9374 4.875C16.6249 4.875 17.3124 5.125 17.9374 5.5625C18.4999 6 18.9999 6.5625 19.2499 7.1875C19.4999 7.8125 19.5624 8.5625 19.4374 9.25ZM8.9374 12.75L7.2499 11.75V7.0625C7.2499 6.375 7.4374 5.625 7.8124 5.0625C8.1874 4.4375 8.7499 4 9.3749 3.6875C9.9999 3.375 10.7499 3.25 11.4374 3.375C12.1249 3.4375 12.8124 3.75 13.3749 4.1875C13.3749 4.1875 13.3124 4.25 13.2499 4.25L9.2499 6.5625C9.1249 6.625 9.0624 6.6875 8.9999 6.8125C8.9374 6.9375 8.9374 7 8.9374 7.125V12.75ZM9.8124 10.75L11.9999 9.5L14.1874 10.75V13.25L11.9999 14.5L9.8124 13.25V10.75Z"></path></svg>
                        <?php esc_html_e('DALL-E', 'magicai-wp'); ?>
                    </label>
                    <input name="type" type="radio" value="sd" id="sd" <?php if ( isset($atts['hide']) && $atts['hide'] == 'dalle') { echo 'checked'; } ?>>
                    <label for="sd">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 2C17.5222 2 22 5.97778 22 10.8889C22 13.9556 19.5111 16.4444 16.4444 16.4444H14.4778C13.5556 16.4444 12.8111 17.1889 12.8111 18.1111C12.8111 18.5333 12.9778 18.9222 13.2333 19.2111C13.5 19.5111 13.6667 19.9 13.6667 20.3333C13.6667 21.2556 12.9 22 12 22C6.47778 22 2 17.5222 2 12C2 6.47778 6.47778 2 12 2ZM10.8111 18.1111C10.8111 16.0843 12.451 14.4444 14.4778 14.4444H16.4444C18.4065 14.4444 20 12.851 20 10.8889C20 7.1392 16.4677 4 12 4C7.58235 4 4 7.58235 4 12C4 16.19 7.2226 19.6285 11.324 19.9718C10.9948 19.4168 10.8111 18.7761 10.8111 18.1111ZM7.5 12C6.67157 12 6 11.3284 6 10.5C6 9.67157 6.67157 9 7.5 9C8.32843 9 9 9.67157 9 10.5C9 11.3284 8.32843 12 7.5 12ZM16.5 12C15.6716 12 15 11.3284 15 10.5C15 9.67157 15.6716 9 16.5 9C17.3284 9 18 9.67157 18 10.5C18 11.3284 17.3284 12 16.5 12ZM12 9C11.1716 9 10.5 8.32843 10.5 7.5C10.5 6.67157 11.1716 6 12 6C12.8284 6 13.5 6.67157 13.5 7.5C13.5 8.32843 12.8284 9 12 9Z"></path></svg>
                        <?php esc_html_e('Stable Diffusion', 'magicai-wp'); ?>
                    </label>
                </div>

                <div class="form-field inline">
                    <label for="prompt"><?php esc_html_e( 'Start with a detailed description', 'magicai-wp' ); ?> <span class="suprise-me"><?php esc_html_e( 'Suprise me', 'magicai-wp' );?> 🪄</span></label>
                    <input type="text" name="prompt" id="prompt" required placeholder="<?php echo esc_attr( magicai_helper()->image_generator_prompt_example() ); ?>">
                    <button class="btn" type="submit"><?php esc_html_e( 'Generate', 'magicai-wp' ) ?></button>
                </div>
                
                <div class="form-field image-uploads dall-e">
                    <label class="media-uploader" for="image"><?php esc_html_e( 'or', 'magicai-wp' ); ?> <em><?php esc_attr_e( 'upload an image', 'magicai-wp' ); ?></em></label>
                    <input type="hidden" name="image" id="image">
            
                    <label class="media-uploader" for="mask" style="display:none"><?php esc_html_e( 'upload an mask image', 'magicai-wp' ); ?></label>
                    <input type="hidden" name="mask" id="mask">
                
                    <div class="variation-wrapper" style="display:none">
                        <input name="variation" type="checkbox" value="variation" id="variation" checked>
                        <label for="variation"><?php esc_html_e( 'use image as a create varitation', 'magicai-wp' ); ?></label>
                    </div>
                </div>

                <?php MagicAI_Shortcodes::instance()->form_data($atts ?? array()); ?>

                <div class="advanced-options-wrapper">
                    <div class="advanced-options sd">
                        <div class="advanced-options-trigger">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M2 18H9V20H2V18ZM2 11H11V13H2V11ZM2 4H22V6H2V4ZM20.674 13.0251L21.8301 12.634L22.8301 14.366L21.914 15.1711C21.9704 15.4386 22 15.7158 22 16C22 16.2842 21.9704 16.5614 21.914 16.8289L22.8301 17.634L21.8301 19.366L20.674 18.9749C20.2635 19.3441 19.7763 19.6295 19.2391 19.8044L19 21H17L16.7609 19.8044C16.2237 19.6295 15.7365 19.3441 15.326 18.9749L14.1699 19.366L13.1699 17.634L14.086 16.8289C14.0296 16.5614 14 16.2842 14 16C14 15.7158 14.0296 15.4386 14.086 15.1711L13.1699 14.366L14.1699 12.634L15.326 13.0251C15.7365 12.6559 16.2237 12.3705 16.7609 12.1956L17 11H19L19.2391 12.1956C19.7763 12.3705 20.2635 12.6559 20.674 13.0251ZM18 18C19.1046 18 20 17.1046 20 16C20 14.8954 19.1046 14 18 14C16.8954 14 16 14.8954 16 16C16 17.1046 16.8954 18 18 18Z"></path></svg>
                            <span><?php esc_html_e( 'Advanced Options' ); ?></span>
                        </div>
                    </div>
                    
                    <div class="form-field image-uploads sd">
                        <label class="media-uploader" for="sd-image"><?php esc_html_e( 'or', 'magicai-wp' ); ?> <em><?php esc_attr_e( 'upload an image', 'magicai-wp' ); ?></em> <?php esc_html_e( 'for image edits or upscale', 'magicai-wp' ); ?></label>
                        <input type="hidden" name="sd-image" id="sd-image">
                        <button type="button" class="sd-upscale">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path d="M21 3C21.5523 3 22 3.44772 22 4V20C22 20.5523 21.5523 21 21 21H3C2.44772 21 2 20.5523 2 20V4C2 3.44772 2.44772 3 3 3H21ZM20 5H4V19H20V5ZM13 17V15H16V12H18V17H13ZM11 7V9H8V12H6V7H11Z"></path></svg>
                            <?php esc_html_e( 'upscale the image?', 'magicai-wp' ); ?>
                        </button>
                    </div>
                </div>

                <div class="advanced-options-form sd">
                    <div class="form-field">
                        <label for="style_preset">
                            <?php magicai_helper()->label_help_tip( 'Pass in a style preset to guide the image model towards a particular style.' ); ?>
                            <?php esc_html_e( 'Style', 'magicai-wp' ); ?>
                        </label>
                        <select name="style_preset" id="style_preset">
                                <option value="" selected><?php esc_html_e( 'None', 'magicai-wp' ); ?></option>
                                <option value="3d-model"><?php esc_html_e( '3D Model', 'magicai-wp' ); ?></option>
                                <option value="analog-film"><?php esc_html_e( 'Analog Film', 'magicai-wp' ); ?></option>
                                <option value="anime"><?php esc_html_e( 'Anime', 'magicai-wp' ); ?></option>
                                <option value="cinematic"><?php esc_html_e( 'Cinematic', 'magicai-wp' ); ?></option>
                                <option value="comic-book"><?php esc_html_e( 'Comic Book', 'magicai-wp' ); ?></option>
                                <option value="digital-art"><?php esc_html_e( 'Digital Art', 'magicai-wp' ); ?></option>
                                <option value="enhance"><?php esc_html_e( 'Enhance', 'magicai-wp' ); ?></option>
                                <option value="fantasy-art"><?php esc_html_e( 'Fantasy Art', 'magicai-wp' ); ?></option>
                                <option value="isometric"><?php esc_html_e( 'Isometric', 'magicai-wp' ); ?></option>
                                <option value="line-art"><?php esc_html_e( 'Line Art', 'magicai-wp' ); ?></option>
                                <option value="low-poly"><?php esc_html_e( 'Low Poly', 'magicai-wp' ); ?></option>
                                <option value="modeling-compound"><?php esc_html_e( 'Modeling Compound', 'magicai-wp' ); ?></option>
                                <option value="neon-punk"><?php esc_html_e( 'Neon Punk', 'magicai-wp' ); ?></option>
                                <option value="origami"><?php esc_html_e( 'Origami', 'magicai-wp' ); ?></option>
                                <option value="photographic"><?php esc_html_e( 'Photographic', 'magicai-wp' ); ?></option>
                                <option value="pixel-art"><?php esc_html_e( 'Pixel Art', 'magicai-wp' ); ?></option>
                                <option value="tile-texture"><?php esc_html_e( 'Tile Texture', 'magicai-wp' ); ?></option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label for="mood">
                            <?php magicai_helper()->label_help_tip( 'Select image mood.' ); ?>
                            <?php esc_html_e( 'Mood', 'magicai-wp' ); ?>
                        </label>
                        <select name="mood" id="mood">
                            <option value="" selected="selected"><?php esc_html_e( 'None', 'magicai-wp' ); ?></option>
                            <option value="aggressive"><?php esc_html_e('Aggressive', 'magicai-wp'); ?></option>
                            <option value="angry"><?php esc_html_e('Angry', 'magicai-wp'); ?></option>
                            <option value="boring"><?php esc_html_e('Boring', 'magicai-wp'); ?></option>
                            <option value="bright"><?php esc_html_e('Bright', 'magicai-wp'); ?></option>
                            <option value="calm"><?php esc_html_e('Calm', 'magicai-wp'); ?></option>
                            <option value="cheerful"><?php esc_html_e('Cheerful', 'magicai-wp'); ?></option>
                            <option value="chilling"><?php esc_html_e('Chilling', 'magicai-wp'); ?></option>
                            <option value="colorful"><?php esc_html_e('Colorful', 'magicai-wp'); ?></option>
                            <option value="dark"><?php esc_html_e('Dark', 'magicai-wp'); ?></option>
                            <option value="neutral"><?php esc_html_e('Neutral', 'magicai-wp'); ?></option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label for="sampler">
                            <?php magicai_helper()->label_help_tip( 'Which sampler to use for the diffusion process. If this value is omitted we\'ll automatically select an appropriate sampler for you.' ); ?>
                            <?php esc_html_e( 'Sampler', 'magicai-wp' ); ?>
                        </label>
                        <select name="sampler" id="sampler">
                            <option value="" selected="selected"><?php esc_html_e( 'None', 'magicai-wp' ); ?></option>
                            <option value="DDIM"><?php esc_html_e('DDIM', 'magicai-wp'); ?></option>
                            <option value="DDPM"><?php esc_html_e('DDPM', 'magicai-wp'); ?></option>
                            <option value="K_DPMPP_2M"><?php esc_html_e('K_DPMPP_2M', 'magicai-wp'); ?></option>
                            <option value="K_DPM_2"><?php esc_html_e('K_DPM_2', 'magicai-wp'); ?></option>
                            <option value="K_DPM_2_ANCESTRAL"><?php esc_html_e('K_DPM_2_ANCESTRAL', 'magicai-wp'); ?></option>
                            <option value="K_EULER"><?php esc_html_e('K_EULER', 'magicai-wp'); ?></option>
                            <option value="K_EULER_ANCESTRAL"><?php esc_html_e('K_EULER_ANCESTRAL', 'magicai-wp'); ?></option>
                            <option value="K_HEUN"><?php esc_html_e('K_HEUN', 'magicai-wp'); ?></option>
                            <option value="K_LMS"><?php esc_html_e('K_LMS', 'magicai-wp'); ?></option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label for="clip_guidance_preset">
                            <?php magicai_helper()->label_help_tip( 'Clip Guidance Preset' ); ?>
                            <?php esc_html_e( 'Clip Guidance Preset', 'magicai-wp' ); ?>
                        </label>
                        <select name="clip_guidance_preset" id="clip_guidance_preset">
                            <option value="" selected="selected"><?php esc_html_e( 'None', 'magicai-wp' ); ?></option>
                            <option value="FAST_BLUE"><?php esc_html_e('FAST BLUE', 'magicai-wp'); ?></option>
                            <option value="FAST_GREEN"><?php esc_html_e('FAST GREEN', 'magicai-wp'); ?></option>
                            <option value="SIMPLE"><?php esc_html_e('SIMPLE', 'magicai-wp'); ?></option>
                            <option value="SLOW"><?php esc_html_e('SLOW', 'magicai-wp'); ?></option>
                            <option value="SLOWER"><?php esc_html_e('SLOWER', 'magicai-wp'); ?></option>
                            <option value="SLOWEST"><?php esc_html_e('SLOWEST', 'magicai-wp'); ?></option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label for="seed">
                            <?php magicai_helper()->label_help_tip( 'Random noise seed (omit this option or use 0 for a random seed).' ); ?>
                            <?php esc_html_e( 'Seed', 'magicai-wp' ); ?>
                        </label>
                        <input name="seed" id="seed" type="number" min="0" max="4294967295" value="0">
                    </div>
                    <div class="form-field">
                        <label for="steps">
                            <?php magicai_helper()->label_help_tip( 'Number of diffusion steps to run.' ); ?>
                            <?php esc_html_e( 'Steps', 'magicai-wp' ); ?>
                        </label>
                        <input name="steps" id="steps" type="number" min="10" max="50" value="10">
                    </div>
                    <div class="form-field">
                        <label for="image_resolution">
                            <?php magicai_helper()->label_help_tip( 'Which sampler to use for the diffusion process. If this value is omitted we\'ll automatically select an appropriate sampler for you.' ); ?>
                            <?php esc_html_e( 'Image Resolution', 'magicai-wp' ); ?>
                        </label>
                        <select name="image_resolution" id="image_resolution">
                            <option value="" selected=""><?php esc_html_e('Default', 'magicai-wp'); ?></option>
                            <option value="640x1536">640 x 1536</option>
                            <option value="768x1344">768 x 1344</option>
                            <option value="832x1216">832 x 1216</option>
                            <option value="896x1152">896 x 1152</option>
                            <option value="1024x1024">1024 x 1024</option>
                            <option value="1152x896">1152 x 896</option>
                            <option value="1216x832">1216 x 832</option>
                            <option value="1344x768">1344 x 768</option>
                            <option value="1536x640">1536 x 640</option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label for="negative_prompt">
                            <?php magicai_helper()->label_help_tip( 'That allows the user to specify what he doesn\'t want to see, without any extra input.' ); ?>
                            <?php esc_html_e( 'Negative Prompt', 'magicai-wp' ); ?>
                        </label>
                        <input type="text" name="negative_prompt" id="negative_prompt">
                    </div>
                </div>

            </form>
        </div>

        <div class="generator-result">
            <div class="gallery">
                <?php magicai_helper()->get_attachment_images( $atts ?? array() ); ?>
            </div>
        </div>

    </div>
</div>