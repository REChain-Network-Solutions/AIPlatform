<div id="tab-code-generator" class="magicai-settings-form-page">
    <div class="magicai-generator-wrapper">

        <div class="generator-form">
            <form action="" class="magicai-generator code-generator magicai-form">
                <div class="form-field">
                    <label for="title">
                        <?php magicai_helper()->label_help_tip( 'Please provide details about the code you want.' ); ?>
                        <?php esc_html_e( 'Describe What Kind of Code You Need', 'magicai-wp' ); ?>
                    </label>
                    <textarea type="text" name="title" id="title" rows="10" placeholder="<?php esc_attr_e( 'Describe What Kind of Code You Need', 'magicai-wp' ); ?>" required></textarea>
                </div>
                
                <div class="form-field">
                    <label for="code">
                        <?php magicai_helper()->label_help_tip( 'Specify the programming language for the code. Example: PHP, WordPress' ); ?>
                        <?php esc_html_e( 'Code Language or Platform (Wordpress, JS, PHP, Go, etc.)', 'magicai-wp' ); ?>
                    </label>
                    <input type="text" name="code" id="code" placeholder="<?php esc_attr_e( 'WordPress, PHP' ); ?>" required>
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