<?php

if ( !defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * MagicAI_Admin_Page_ChatBot class for handling the MagicAI admin page.
 *
 * This class extends the MagicAI_Admin_Page class to create a specific admin page for MagicAI.
 *
 * @since 1.0.0
 */
class MagicAI_Admin_Page_ChatBot extends MagicAI_Admin_Page {

	/**
     * Constructor method for initializing the MagicAI admin page.
     *
     * @method __construct
     */
	public function __construct() {

		$this->id = 'magicai-chatbot';
		$this->page_title = esc_html__( 'AI ChatBot', 'magicai-wp' );
		$this->menu_title = esc_html__( 'AI ChatBot', 'magicai-wp' );
		$this->parent = 'magicai';
		$this->position = '90';
        
		parent::__construct();

		$this->options = get_option('magicai_chatbot_settings', array() );

		if ( isset( $this->options['status'] ) ) {

			$is_user_logged_in = $this->options['is_user_logged_in'];

			if ( ! $this->options['template'] ) {
				return;
			}
			$this->bot_name = get_post_meta( $this->options['template'], '_name', true );
			$this->bot_role = get_post_meta( $this->options['template'], '_role', true );

			if ( ( $is_user_logged_in && is_user_logged_in() ) || !$is_user_logged_in ) {
				if ( $this->options['status'] == 'wp' ) {
					add_action('admin_footer', [$this, 'chatbot_widget']);
				} elseif ( $this->options['status'] == 'frontend' ) {
					add_action('wp_footer', [$this, 'chatbot_widget']);
				} elseif ( $this->options['status'] == 'both' ){
					add_action('admin_footer', [$this, 'chatbot_widget']);
					add_action('wp_footer', [$this, 'chatbot_widget']);
				}
			}
		}
	}

	/**
     * Display the content for the MagicAI admin page.
     *
     * @method display
     * @return void
     */
	public function display() {
		
		include_once __DIR__ . '/views/chatbot.php';

	}

	/**
     * Save method for handling data saving on the MagicAI admin page.
     *
     * @method save
     * @return void
     */
	public function save() {

	}

	public function chatbot_widget() {
		$_image = get_post_meta( $this->options['template'], '_image', true );
		?>
		<style>
			:root{
				--magicai-chatbot--trigger-bg: <?php echo get_post_meta( $this->options['template'], '_trigger_bg', true )?>;
				--magicai-chatbot--chat-bg: <?php echo get_post_meta( $this->options['template'], '_chat_bg', true )?>;
				--magicai-chatbot--message-ai-bg: <?php echo get_post_meta( $this->options['template'], '_message_ai_bg', true )?>;
				--magicai-chatbot--message-ai-color: <?php echo get_post_meta( $this->options['template'], '_message_ai_color', true )?>;
				--magicai-chatbot--message-bg: <?php echo get_post_meta( $this->options['template'], '_message_bg', true )?>;
				--magicai-chatbot--message-color: <?php echo get_post_meta( $this->options['template'], '_message_color', true )?>;
				--magicai-chatbot--title-color: <?php echo get_post_meta( $this->options['template'], '_title_color', true )?>;
				--magicai-chatbot--subitle-color: <?php echo get_post_meta( $this->options['template'], '_subtitle_color', true )?>;
				--magicai-chatbot--title-border-color: <?php echo get_post_meta( $this->options['template'], '_title_border_color', true )?>;
				--magicai-chatbot--input-bg: <?php echo get_post_meta( $this->options['template'], '_input_bg', true )?>;
				--magicai-chatbot--input-color: <?php echo get_post_meta( $this->options['template'], '_input_color', true )?>;
				--magicai-chatbot--input-border-color: <?php echo get_post_meta( $this->options['template'], '_input_border_color', true )?>;
				--magicai-chatbot--btn-color: <?php echo get_post_meta( $this->options['template'], '_btn_color', true )?>;
				--magicai-chatbot--width: <?php echo get_post_meta( $this->options['template'], '_width', true ) ? get_post_meta( $this->options['template'], '_width', true ) : '360px'?>;
				--magicai-chatbot--height: <?php echo get_post_meta( $this->options['template'], '_height', true ) ? get_post_meta( $this->options['template'], '_height', true ) : '420px'?>;
			}
		</style>
		<div id="magicai-chatbot-widget" class="<?php echo esc_attr( $this->options['position'] ); ?>">
			<div class="magicai-chatbot-widget--chat">
				<div class="magicai-chatbot-widget--header">
					<div class="title">
						<?php echo esc_html($this->bot_name); ?><br>
						<span><?php echo esc_html($this->bot_role); ?></span>
					</div>
					<div class="message-list">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M21 17.9995V19.9995H3V17.9995H21ZM6.59619 3.90332L8.01041 5.31753L4.82843 8.49951L8.01041 11.6815L6.59619 13.0957L2 8.49951L6.59619 3.90332ZM21 10.9995V12.9995H12V10.9995H21ZM21 3.99951V5.99951H12V3.99951H21Z"></path></svg>
					</div>
				</div>
				<div class="loader">
					<div class="bar"></div>
					<div class="bar" style="width:100%"></div>
					<div class="bar" style="width:40%"></div>
				</div>
				<div class="magicai-chatbot-widget--message-list"></div>
				<button type="button" class="magicai-btn start-new-chat"><?php esc_html_e( 'Start New Chat', 'magicai-wp' ); ?></button>
				<div class="magicai-chatbot-widget--messages"></div>
				<form action="">
					<input type="text" name="prompt" id="prompt" placeholder="<?php esc_attr_e( 'Type and hit...', 'magicai-wp' ) ?>" required autocomplete="off">
					<button type="submit" class="start">
						<svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" width="18" height="18"><path d="M3.5 1.3457C3.58425 1.3457 3.66714 1.36699 3.74096 1.4076L22.2034 11.562C22.4454 11.695 22.5337 11.9991 22.4006 12.241C22.3549 12.3241 22.2865 12.3925 22.2034 12.4382L3.74096 22.5925C3.499 22.7256 3.19497 22.6374 3.06189 22.3954C3.02129 22.3216 3 22.2387 3 22.1544V1.8457C3 1.56956 3.22386 1.3457 3.5 1.3457ZM5 4.38261V11.0001H10V13.0001H5V19.6175L18.8499 12.0001L5 4.38261Z"></path></svg>
					</button>
				</form>
			</div>
			<div class="magicai-chatbot-widget--trigger">
				<img src="<?php esc_attr_e( $_image ); ?>" class="<?php echo esc_attr(  strpos( basename($_image), '.svg' ) !== false ? 'svg' : '' ); ?>" alt="ChatBot">
			</div>
		</div>
		<?php

	}
}
new MagicAI_Admin_Page_ChatBot;
