<div id="tab-custom-web-generator" class="magicai-settings-form-page">
    <div class="magicai-generator-wrapper">

        <div class="generator-form">
            <form action="" class="magicai-generator custom-web-generator magicai-form">
                <div class="form-field">
                    <label for="prompt">
                        <?php magicai_helper()->label_help_tip( 'Write in detail what you want according to your preferences.' ); ?>
                        <?php esc_html_e( 'Describe What You Need', 'magicai-wp' ); ?>
                    </label>
                    <textarea type="text" name="prompt" id="prompt" rows="10" placeholder="<?php esc_attr_e( 'I can search the entire web for you and generate a result based on those results.', 'magicai-wp' ); ?>" required></textarea>
                </div>

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