<div class="wrap magicai-page-wrap">
    <span class="magicai-loader"></span>
    <h1><?php echo esc_html( get_admin_page_title() ); ?></h1>
    <p><?php esc_html_e( 'Easily transform audio into written text.', 'magicai-wp' ); ?></p>

    <div class="magicai-speech-to-text">
        
        <div class="magicai-generator-wrapper">

            <div class="generator-form">
                <form action="" class="magicai-generator speech-to-text magicai-form">
                    <div class="form-field file">
                        <label for="prompt">
                            <?php magicai_helper()->label_help_tip( 'Upload your audio file and press the generate button.' ); ?>
                            <?php esc_html_e( 'Upload an Audio File (mp3, mp4, mpeg, mpga, m4a, wav, and webm)(Max: 25Mb)', 'magicai-wp' ); ?>
                        </label>
                        <label class="media-uploader" for="file"><?php esc_html_e( 'Select File', 'magicai-wp' ); ?></label>
                        <input type="hidden" name="file" id="file">
                    </div>

                    <div class="form-field">
                        <button class="btn" type="submit"><?php esc_html_e( 'Generate', 'magicai-wp' ) ?></button>
                    </div>

                </form>
            </div>

            <?php $content = magicai_helper()->get_documents_data( 'transcribe' ); ?>

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
</div>