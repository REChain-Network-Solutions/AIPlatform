<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

use Firebase\JWT\JWT;
use Firebase\JWT\Key;

#[AllowDynamicProperties]
class MagicAI_Shortcodes {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Shortcodes The single instance of the class.
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
	 * @return MagicAI_Shortcodes An instance of the class.
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
       
        $this->init();

    }

    /**
	 * Initialize
	 *
	 * Load the files required to run the plugin.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function init() {

		$this->hooks();

    }

    /**
     * Hooks Registration
     *
     * This function registers WordPress action and filter hooks for the MagicAI plugin.
     * It allows you to define custom actions and filters to extend or modify the plugin's functionality.
     *
     * @since 1.0.0
     *
     * @access public
     */
	public function hooks() {

        add_action( 'wp_head', [$this, 'loader'] );
        add_shortcode( 'magicai-generator', [$this, 'create_generator_shortcode'] );
        add_shortcode( 'magicai-image-generator', [$this, 'create_image_generator_shortcode'] );
        add_shortcode( 'magicai-chat', [$this, 'create_chat_shortcode'] );
        add_shortcode( 'magicai-logs', [$this, 'create_logs_shortcode'] );
        add_shortcode( 'magicai-documents', [$this, 'create_documents_shortcode'] );

        add_action( 'wp_ajax_magicai_get_document_modal', [$this, 'get_document_modal'] );

	}

    // Create Shortcode magicai-generator
    // Shortcode: [magicai-generator type="custom" model="gpt-3.5-turbo" language="en-US" n="1" temperature="0.75" tone="Professional" is_user_logged_in="true" api_key=""]
    function create_generator_shortcode($atts) {

        // Attributes
        $atts = shortcode_atts(
            array(
                'type' => 'custom',
                'model' => magicai_helper()->get_option( 'openai_model', 'gpt-3.5-turbo-instruct' ),
                'temperature' => magicai_helper()->get_option( 'openai_temperature', 0.75 ),
                'frequency_penalty' => magicai_helper()->get_option( 'openai_frequency_penalty', 0 ),
                'presence_penalty' => magicai_helper()->get_option( 'openai_presence_penalty', 0.6 ),
                'max_tokens' => magicai_helper()->get_option( 'openai_max_tokens', 800 ),
                'api_key' => '',
                'is_user_logged_in' => 'true',
            ),
            $atts,
            'magicai-generator'
        );

        // Attributes in var
        $type = $atts['type'];
        $is_user_logged_in = $atts['is_user_logged_in'];

        if ( $is_user_logged_in == 'true' && !is_user_logged_in() ) {
            return esc_html__('You must login first!', 'magicai-wp');
        }

        ob_start();

        // Codes
        switch( $type ) {
            case 'custom': {
                include MAGICAI_PATH . 'includes/admin/views/tabs/custom.php';
                break;
            }
            case 'code': {
                include MAGICAI_PATH . 'includes/admin/views/tabs/code.php';
                break;
            }
            case 'product': {
                include MAGICAI_PATH . 'includes/admin/views/tabs/product.php';
                break;
            }
            case 'post': {
                include MAGICAI_PATH . 'includes/admin/views/tabs/post.php';
                break;
            }
            case 'post_youtube_video': {
                include MAGICAI_PATH . 'includes/admin/views/tabs/yt-post.php';
                break;
            }
            case 'post_rss': {
                include MAGICAI_PATH . 'includes/admin/views/tabs/rss-post.php';
                break;
            }
        }

        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;

    }

    // Create Shortcode magicai-image-generator
    // Shortcode: [magicai-image-generator hide="sd" dalle_api_key="" dalle_n="1" dalle_size="256x256" sd_api_key="" sd_model="stable-diffusion-512-v2-1" sd_n="1" sd_size="512" user_filter="false" is_user_logged_in="true"]
    function create_image_generator_shortcode($atts) {

        // Attributes
        $atts = shortcode_atts(
            array(
                'hide' => '', // dalle, sd
                'dalle_api_key' => '',
                'dalle_n' => magicai_helper()->get_option( 'dalle_n', 1 ),
                'dalle_size' => magicai_helper()->get_option( 'dalle_size', '256x256' ),
                'sd_api_key' => '',
                'sd_model' => magicai_helper()->get_option( 'sd_model', 'stable-diffusion-512-v2-1' ),
                'sd_n' => magicai_helper()->get_option( 'sd_n', 1 ),
                'sd_size' => magicai_helper()->get_option( 'sd_size', 512 ),
                'user_filter' => 'true', // true, false or user_id
                'is_user_logged_in' => 'true', // true, false
            ),
            $atts,
            'magicai-image-generator'
        );

        // Attributes in var
        $is_user_logged_in = $atts['is_user_logged_in'];

        if ( $is_user_logged_in == 'true' && !is_user_logged_in() ) {
            return esc_html__('You must login first!', 'magicai-wp');
        }

        ob_start();

        // Codes
        include MAGICAI_PATH . 'includes/admin/views/tabs/image.php';

        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;

    }

    // Create Shortcode magicai-chat
    // Shortcode: [magicai-chat user_filter="true" is_user_logged_in="true"]
    function create_chat_shortcode($atts) {

        // Attributes
        $atts = shortcode_atts(
            array(
                'user_filter' => 'true', // true, false or user_id
                'is_user_logged_in' => 'true', // true, false
            ),
            $atts,
            'magicai-chat'
        );

        // Attributes in var
        $is_user_logged_in = $atts['is_user_logged_in'];

        if ( $is_user_logged_in == 'true' && !is_user_logged_in() ) {
            return esc_html__('You must login first!', 'magicai-wp');
        }

        ob_start();

        // Codes
        include MAGICAI_PATH . 'includes/admin/views/chat.php';

        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;

    }

    // Create Shortcode magicai-logs
    // Shortcode: [magicai-logs user_filter="true" count="5"]
    function create_logs_shortcode($atts) {

        // Attributes
        $atts = shortcode_atts(
            array(
                'user_filter' => 'true', // true, false or user_id
                'count' => 5,
            ),
            $atts,
            'magicai-logs'
        );

        if ( !is_user_logged_in() ) {
            return;
        }

        ob_start();

        // Codes
        ?>
        <div class="magicai-dashboard--logs">
            <?php MagicAI_Logs::instance()->get_last_logs( intval($atts['count']), $atts ); ?>
        </div>
        <?php

        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;

    }
    
    // Create Shortcode magicai-documents
    // Shortcode: [magicai-documents user_filter="true"]
    function create_documents_shortcode($atts) {

        // Attributes
        $atts = shortcode_atts(
            array(
                'user_filter' => 'true', // true, false or user_id
            ),
            $atts,
            'magicai-documents'
        );

        if ( !is_user_logged_in() ) {
            return;
        }

        // Attributes in var
        $user_filter = $atts['user_filter'];

        ob_start();

        // Codes
        $args = [
            'post_type' => 'magicai-documents',
            'posts_per_page' => -1,
        ];

        if ( !empty( $user_filter ) || $user_filter == 'true' ) {
            $args['meta_key'] = '_magicai_userid';
            $args['meta_value'] = $user_filter == 'true' ? get_current_user_id() : $user_filter;
        }

        $posts = get_posts( $args );

        ?>
        <table class="magicai-sc--documents-table">
            <thead>
                <tr>
                    <th><?php esc_html_e('Title', 'magicai-wp'); ?></th>
                    <th><?php esc_html_e('Type', 'magicai-wp'); ?></th>
                    <th><?php esc_html_e('Content', 'magicai-wp'); ?></th>
                    <th><?php esc_html_e('Date', 'magicai-wp'); ?></th>
                </tr>
            </thead>
            <tbody>
            <?php
                foreach ( $posts as $post ) {
                    printf( 
                        '<tr class="type-magicai-documents">
                            <td>%s</td>
                            <td class="type column-type">%s</td>
                            <td><span class="magicai-sc--documents-open" data-postid="%s">%s</span></td>
                            <td>%s</td>
                        </tr>',
                        $post->post_title,
                        $this->documents_type_style( get_post_meta( $post->ID, '_magicai_doc_type', true ) ),
                        $post->ID,
                        esc_html__( 'Open Result', 'magicai-wp' ),
                        $post->post_date,
                    );
                }
            ?>
            </tbody>
        </table>
        <?php

        // var_dump($posts);


        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;

    }

    function loader() {
        ?><span class="magicai-loader"></span><?php
    }

    function form_data( $atts = null ) {

        if ( !$atts ) return;

        $jwt = JWT::encode($atts, magicai_helper()->magicai_salt(), 'HS256');
       
        printf( 
            '<input id="atts" name="atts" type="hidden" value="%s">',
            esc_attr($jwt)
        );

    }

    function documents_type_style( $type ) {
        if ( $type == 'transcribe' ) {
            return sprintf( 
                '<span class="transcribe">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M11.9998 3C10.3429 3 8.99976 4.34315 8.99976 6V10C8.99976 11.6569 10.3429 13 11.9998 13C13.6566 13 14.9998 11.6569 14.9998 10V6C14.9998 4.34315 13.6566 3 11.9998 3ZM11.9998 1C14.7612 1 16.9998 3.23858 16.9998 6V10C16.9998 12.7614 14.7612 15 11.9998 15C9.23833 15 6.99976 12.7614 6.99976 10V6C6.99976 3.23858 9.23833 1 11.9998 1ZM3.05469 11H5.07065C5.55588 14.3923 8.47329 17 11.9998 17C15.5262 17 18.4436 14.3923 18.9289 11H20.9448C20.4837 15.1716 17.1714 18.4839 12.9998 18.9451V23H10.9998V18.9451C6.82814 18.4839 3.51584 15.1716 3.05469 11Z"></path></svg>
                    %s
                </span>', esc_html__( 'Speech to Text', 'magicai-wp' )
            );
        } elseif ( $type == 'voiceover' ) {
            return sprintf( 
                '<span class="voiceover">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M16.9337 8.96494C16.426 5.03562 13.0675 2 9 2 4.58172 2 1 5.58172 1 10 1 11.8924 1.65707 13.6313 2.7555 15.0011 3.56351 16.0087 4.00033 17.1252 4.00025 18.3061L4 22H13L13.001 19H15C16.1046 19 17 18.1046 17 17V14.071L18.9593 13.2317C19.3025 13.0847 19.3324 12.7367 19.1842 12.5037L16.9337 8.96494ZM3 10C3 6.68629 5.68629 4 9 4 12.0243 4 14.5665 6.25141 14.9501 9.22118L15.0072 9.66262 16.5497 12.0881 15 12.7519V17H11.0017L11.0007 20H6.00013L6.00025 18.3063C6.00036 16.6672 5.40965 15.114 4.31578 13.7499 3.46818 12.6929 3 11.3849 3 10ZM21.1535 18.1024 19.4893 16.9929C20.4436 15.5642 21 13.8471 21 12.0001 21 10.153 20.4436 8.4359 19.4893 7.00722L21.1535 5.89771C22.32 7.64386 23 9.74254 23 12.0001 23 14.2576 22.32 16.3562 21.1535 18.1024Z"></path></svg>
                    %s
                </span>', esc_html__( 'Voiceover', 'magicai-wp' )
            );
        } elseif ( $type == 'code' ) {
            return sprintf( 
                '<span class="code">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M24 12L18.3431 17.6569L16.9289 16.2426L21.1716 12L16.9289 7.75736L18.3431 6.34315L24 12ZM2.82843 12L7.07107 16.2426L5.65685 17.6569L0 12L5.65685 6.34315L7.07107 7.75736L2.82843 12ZM9.78845 21H7.66009L14.2116 3H16.3399L9.78845 21Z"></path></svg>
                    %s
                </span>', esc_html__( 'Code', 'magicai-wp' )
            );
        } elseif ( $type == 'post' ) {
            return sprintf( 
                '<span class="post">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M20 22H4C3.44772 22 3 21.5523 3 21V3C3 2.44772 3.44772 2 4 2H20C20.5523 2 21 2.44772 21 3V21C21 21.5523 20.5523 22 20 22ZM19 20V4H5V20H19ZM7 6H11V10H7V6ZM7 12H17V14H7V12ZM7 16H17V18H7V16ZM13 7H17V9H13V7Z"></path></svg>
                    %s
                </span>', esc_html__( 'Post', 'magicai-wp' )
            );
        } elseif ( $type == 'product' ) {
            return sprintf( 
                '<span class="product">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M4.00436 6.41662L0.761719 3.17398L2.17593 1.75977L5.41857 5.00241H20.6603C21.2126 5.00241 21.6603 5.45012 21.6603 6.00241C21.6603 6.09973 21.6461 6.19653 21.6182 6.28975L19.2182 14.2898C19.0913 14.7127 18.7019 15.0024 18.2603 15.0024H6.00436V17.0024H17.0044V19.0024H5.00436C4.45207 19.0024 4.00436 18.5547 4.00436 18.0024V6.41662ZM6.00436 7.00241V13.0024H17.5163L19.3163 7.00241H6.00436ZM5.50436 23.0024C4.67593 23.0024 4.00436 22.3308 4.00436 21.5024C4.00436 20.674 4.67593 20.0024 5.50436 20.0024C6.33279 20.0024 7.00436 20.674 7.00436 21.5024C7.00436 22.3308 6.33279 23.0024 5.50436 23.0024ZM17.5044 23.0024C16.6759 23.0024 16.0044 22.3308 16.0044 21.5024C16.0044 20.674 16.6759 20.0024 17.5044 20.0024C18.3328 20.0024 19.0044 20.674 19.0044 21.5024C19.0044 22.3308 18.3328 23.0024 17.5044 23.0024Z"></path></svg>
                    %s
                </span>', esc_html__( 'Product', 'magicai-wp' )
            );
        } else {
            return sprintf( 
                '<span class="text">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M20 22H4C3.44772 22 3 21.5523 3 21V3C3 2.44772 3.44772 2 4 2H20C20.5523 2 21 2.44772 21 3V21C21 21.5523 20.5523 22 20 22ZM19 20V4H5V20H19ZM7 6H11V10H7V6ZM7 12H17V14H7V12ZM7 16H17V18H7V16ZM13 7H17V9H13V7Z"></path></svg>
                    %s
                </span>', esc_html__( 'Text', 'magicai-wp' )
            );
        }
    }

    function get_document_modal() {

        // sanitize form data
        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';

        if ( empty( $post_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Document does not exists!', 'magicai-wp' ),
            ] );
        }

        $output = get_post_field( 'post_content', $post_id );

        if ( get_post_meta( $post_id, '_magicai_doc_type', true ) == 'code' ) {
            $output = sprintf( 
                '<div class="theme-tomorrow"><pre><code class="line-numbers language-%s">%s</code></pre></div>',
                magicai_helper()->detect_programming_language($output),
                $output
            );
        }


        wp_send_json( [
            'post_title' => esc_html(get_post_field( 'post_title', $post_id )),
            'post_content' => $output,
        ] );
    }


}
MagicAI_Shortcodes::instance();
