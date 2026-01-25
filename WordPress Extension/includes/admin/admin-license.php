<?php

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

class MagicAI_License {

    /**
	 * Variables required for the theme updater
	 *
	 * @since 1.0.0
	 * @type string
	 */
    protected $slug = null;

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_License The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_License An instance of the class.
	 */
	public static function instance() {

		if ( is_null( self::$_instance ) ) {
			self::$_instance = new self();
		}
		return self::$_instance;
    }

    /**
	 * Constructor
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function __construct() {

        $this->slug = 'magicai';
        $this->check_license();
        $this->check_domain();

    }

    /**
     * Display the license form and related information in the MagicAI dashboard.
     *
     * @return void
     *
     * @since 1.0.0
     */
    function form() {

        global $wp;
		$url = add_query_arg( $_GET, $wp->request );
		$status = get_option( $this->slug . '_purchase_code_status', false );
        $message = get_transient( $this->slug . '_license_message' );

        if ( $status == 'valid' ) {
            $bg = '#3dc764';
        } else {
            $bg = '#ffc600';
            if ( $message ) {
                $bg = '#ed3d3d';
            }
        }

        ?>

        <div class="magicai-dashboard--col magicai-dashboard--license" style="--bg:<?php esc_attr_e( $bg ); ?>">
            <h2>
                <svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M12 1L20.2169 2.82598C20.6745 2.92766 21 3.33347 21 3.80217V13.7889C21 15.795 19.9974 17.6684 18.3282 18.7812L12 23L5.6718 18.7812C4.00261 17.6684 3 15.795 3 13.7889V3.80217C3 3.33347 3.32553 2.92766 3.78307 2.82598L12 1ZM12 3.04879L5 4.60434V13.7889C5 15.1263 5.6684 16.3752 6.7812 17.1171L12 20.5963L17.2188 17.1171C18.3316 16.3752 19 15.1263 19 13.7889V4.60434L12 3.04879ZM12 7C13.1046 7 14 7.89543 14 9C14 9.73984 13.5983 10.3858 13.0011 10.7318L13 15H11L10.9999 10.7324C10.4022 10.3866 10 9.74025 10 9C10 7.89543 10.8954 7 12 7Z"></path></svg>
                <?php esc_html_e( 'License', 'magicai-wp' ); ?>
            </h2>
            <?php if ( $status === 'valid' ) : ?>
                <?php
                    printf(
                        '<p>%s <a href="https://magicaidocs-wp.liquid-themes.com/" target="_blank">%s</a> %s <a href="https://liquidthemes.freshdesk.com/support/home" target="_blank">%s</a>.</p>',
                        esc_html__('You can now enjoy MagicAI. Looking for help? Visit', 'magicai-wp'),
                        esc_html__('our help center', 'magicai-wp'),
                        esc_html__('or', 'magicai-wp'),
                        esc_html__('submit a ticket', 'magicai-wp')
                    );
                ?>
            <?php else: ?>
                <?php
                    printf(
                        '<p>%s <a href="https://portal.liquid-themes.com/" target="_blank">portal.liquid-themes.com</a> %s</p>',
                        esc_html__('Go to', 'magicai-wp'),
                        esc_html__('and create your Liquid account before activating the MagicAI.', 'magicai-wp')
                    );
                ?>
                <form action="https://portal.liquid-themes.com/license/activate" method="GET" target="_blank" class="magicai-form">
                    <input type="hidden" name="envato_item_id" value="50173076" />
                    <input type="hidden" name="theme" value="magicai-wp" />
                    <input type="hidden" name="domain" value="<?php echo site_url(); ?>" />
                    <input type="hidden" name="return_url" value="<?php echo admin_url( 'admin.php?page=magicai' ); ?>" />
                    <div class="form-field">
                    <button type="submit" class="btn">
                        <?php esc_html_e( 'Connect to Liquid Portal', 'magicai-wp' ) ?>
                    </button>
                    </div>
                </form>
                <?php if ( $message ) : ?>
                    <div class="magicai-dashboard--license-message">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor" style="vertical-align:text-top"><path d="M12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22ZM11 15V17H13V15H11ZM11 7V13H13V7H11Z"></path></svg>
                        <?php echo $message; ?>
                    </div>
                <?php endif; ?>
            <?php endif; ?>
        </div>

        <?php

    }

    /**
     * Check the license status and update options accordingly.
     *
     * @return void
     *
     * @since 1.0.0
     */
    function check_license() {

        // Check if the necessary parameters are set in the $_GET array
		if ( ! isset( $_GET['liquid_license_status'] ) && ! isset( $_GET['liquid_license_key'] ) && ! isset( $_GET['liquid_license_domain_key'] ) ) {
			return;
		}

        global $wp;
		$url = add_query_arg( $_GET, $wp->request );

		if ( $_GET['liquid_license_status'] === 'valid' ) {

			update_option( $this->slug . '_purchase_code_status', $_GET['liquid_license_status'] );
			update_option( $this->slug . '_purchase_code', $_GET['liquid_license_key'] );
			update_option( $this->slug . '_purchase_code_domain_key', $_GET['liquid_license_domain_key'] );
			update_option( $this->slug . '_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()) );
			set_transient( $this->slug . '_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()), 12 * HOUR_IN_SECONDS );

			$message = '';

		} else {

			if ( isset($_GET['liquid_license_message']) && ! empty( $_GET['liquid_license_message'] ) ) {
				$message_text = $_GET['liquid_license_message'];
			} else {
				$message_text = 'Something went wrong! Contact to support.';
			}

			$message = $message_text;
		}

		set_transient( $this->slug . '_license_message', $message, ( 24 * HOUR_IN_SECONDS ) );

        wp_redirect(admin_url('admin.php?page=magicai'));

	}

    /**
	 * Checks domain expire date.
	 *
	 *
	 * @since 1.0.0
	 */
	function check_domain() {

		if ( !get_option( $this->slug . '_purchase_code_domain' ) ){
			update_option( $this->slug . '_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()) );
			set_transient( $this->slug . '_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()), 12 * HOUR_IN_SECONDS );
		} else {
			if ( false === get_transient( $this->slug . '_purchase_code_domain' ) ) {
				if ( str_replace(array("http://","https://"),"",site_url()) != get_option( $this->slug . '_purchase_code_domain' ) ){
					delete_option( $this->slug . '_purchase_code_status' );
					delete_option( $this->slug . '_purchase_code'  );
					delete_option( $this->slug . '_purchase_code_domain_key' );
					delete_option( $this->slug . '_purchase_code_domain' );
					update_option( $this->slug . '_purchase_code_domain_migrated', 'Migrate detected!' );
                    set_transient( $this->slug . '_license_message', 'Migrate detected. Please Contact Support!', ( 7 * DAY_IN_SECONDS ) );
				} else {
					set_transient( $this->slug . '_purchase_code_domain', str_replace(array("http://","https://"),"",site_url()), 12 * HOUR_IN_SECONDS );
				}
			}
		}

	}

    /**
     * Retrieves the license status from the WordPress options.
     *
     * @return string|false The license status or false if not found.
     *
     * @since 1.0.0
     */
    function get_license_status() {
        $status = get_option( $this->slug . '_purchase_code_status', false );
        return $status;
    }

    /**
     * Handles AJAX request for displaying messages based on license status.
     *
     * @return void
     *
     * @since 1.0.0
     */
    function ajax_message() {
        $status = $this->get_license_status();

        if ( $status !== 'valid' ) {
            wp_send_json([
                'error' => true,
                'message' => 'Please Active The MagicAI'
            ]);
        }
    }


}
MagicAI_License::instance();
