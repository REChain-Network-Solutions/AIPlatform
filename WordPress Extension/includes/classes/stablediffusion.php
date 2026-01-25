<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

use Firebase\JWT\JWT;
use Firebase\JWT\Key;

#[AllowDynamicProperties]
class MagicAI_StableDiffusion {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_StableDiffusion The single instance of the class.
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
	 * @return MagicAI_StableDiffusion An instance of the class.
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

		$this->load_options();
		$this->hooks();
        $this->include_files();

    }

    /**
     * Load Options
     *
     * This function loads the MagicAI plugin options from the WordPress database.
     * It retrieves the settings.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function load_options() {

        // Retrieve the MagicAI settings from the database
        $this->sd_key   = magicai_helper()->get_option( 'sd_key' );
        $this->sd_model = magicai_helper()->get_option( 'sd_model', 'stable-diffusion-512-v2-1' );
        $this->sd_n     = intval( magicai_helper()->get_option( 'sd_n', 1 ) );
        $this->sd_size  = intval( magicai_helper()->get_option( 'sd_size', 512 ) );

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

        add_action( 'wp_ajax_magicai_generate_sd_image', [ $this, 'sendRequest' ] );

	}
    
    /**
     * Include Required Files
     *
     * This function includes the necessary PHP files and dependencies for the MagicAI plugin to function correctly.
     * It ensures that all required files are loaded for proper execution of the plugin's features.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function include_files() {
        
    }

    function sendRequest() {

        MagicAI_License::instance()->ajax_message();

        $opts = array();

        $prompt = !empty( $_POST['prompt'] ) ? sanitize_text_field( $_POST['prompt'] ) : '';
        $image = !empty( $_POST['sd-image'] ) ? intval( $_POST['sd-image'] ) : '';
        $upscale = !empty( $_POST['upscale'] ) ? intval( $_POST['upscale'] ) : 0;

        // Advanced options
        $style_preset = !empty( $_POST['style_preset'] ) ? sanitize_text_field( $_POST['style_preset'] ) : '';
        $mood = !empty( $_POST['mood'] ) ? sanitize_text_field( $_POST['mood'] ) : '';
        $sampler = !empty( $_POST['sampler'] ) ? sanitize_text_field( $_POST['sampler'] ) : '';
        $clip_guidance_preset = !empty( $_POST['clip_guidance_preset'] ) ? sanitize_text_field( $_POST['clip_guidance_preset'] ) : '';
        $seed = !empty( $_POST['seed'] ) ? intval( $_POST['seed'] ) : '';
        $steps = !empty( $_POST['steps'] ) ? intval( $_POST['steps'] ) : 30;
        $image_resolution = !empty( $_POST['image_resolution'] ) ? sanitize_text_field( $_POST['image_resolution'] ) : '';
        $negative_prompt = !empty( $_POST['negative_prompt'] ) ? sanitize_text_field( $_POST['negative_prompt'] ) : '';

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));
                $this->sd_key = !empty($atts->sd_api_key) ? $atts->sd_api_key : $this->sd_key;
                $this->sd_model = !empty($atts->sd_model) ? $atts->sd_model : $this->sd_model;
                $this->sd_n = !empty($atts->sd_n) ? intval($atts->sd_n) : $this->sd_n;
                $this->sd_size = !empty($atts->sd_size) ? intval($atts->sd_size) : $this->sd_size;
            } catch (LogicException $e) {
                wp_send_json( [
                    'error' => true,
                    'message' => $e->getMessage(),
                ] );
            } catch (UnexpectedValueException $e) {
                wp_send_json( [
                    'error' => true,
                    'message' => $e->getMessage(),
                ] );
            }
        }

        // Params
        if ( !empty( $mood ) ) {
            $prompt .= ' ' . $mood . ' mood.';
        }
        $opts['text_prompts'][] = [
            'text' => $prompt,
        ];
        $opts['cfg_scale'] = 7;
        $opts['samples'] = $this->sd_n;
        if ( !empty( $style_preset ) ) {
            $opts['style_preset'] = $style_preset;
        }
        if ( !empty( $sampler ) ) {
            $opts['sampler'] = $sampler;
        }
        if ( !empty( $clip_guidance_preset ) ) {
            $opts['clip_guidance_preset'] = $clip_guidance_preset;
        }
        if ( !empty( $seed ) ) {
            $opts['seed'] = $seed;
        }
        if ( !empty( $steps ) ) {
            $opts['steps'] = $steps;
        }
        if ( !empty( $image_resolution ) ) {
            $image_resolution = explode('x', $image_resolution);
            $opts['width'] = intval($image_resolution[0]);
            $opts['height'] = intval($image_resolution[1]);
        } else {
            $opts['width'] = $this->sd_size;
            $opts['height'] = $this->sd_size;
        }
        if ( !empty( $negative_prompt ) ) {
            $opts['text_prompts'][] = [
              'text' => $negative_prompt,
              'weight' => -1
            ];
        }

        // Types
        if ( empty( $image ) ) { // text2image 
            $type = 'text-to-image';
        } else {
            $type = 'image-to-image';
            if ( $upscale ) {
                $type = 'upscale';
            }
        }

        $response = $this->remote_post( $type, $image, $opts );
        $data = json_decode($response, true);

        $output = '';

        if ($data && isset($data['artifacts']) && is_array($data['artifacts'])) {
            foreach ($data['artifacts'] as $artifact) {
                if ( isset( $artifact['base64'] ) ) {
                    $image_data = base64_decode( $artifact['base64'] );
                    if ( magicai_helper()->get_option('storage', 'wp') == 's3' ) {
                        // Insert image to s3
                        $attachment_id = magicai_helper()->insert_image_to_s3( $image->url, $prompt, $base64 = true );
                        $output .= sprintf( 
                            '<figure class="gallery-item" data-src="%2$s" data-prompt="%3$s" data-name="%4$s" data-id="%5$s">%1$s</figure>',
                            '<img src="'.get_post_field( 'post_content', $attachment_id ).'" class="gallery-img">',
                            get_post_field( 'post_content', $attachment_id ),
                            $prompt,
                            basename( get_post_field( 'post_content', $attachment_id ) ),
                            esc_attr( $attachment_id )
                        );
                    } else {
                        // Insert image to wp media gallery
                        $attachement_id = magicai_helper()->insert_image_to_gallery($image_data, $prompt, true, true);
                        $output .= sprintf( 
                            '<figure class="gallery-item" data-src="%2$s" data-prompt="%3$s" data-name="%4$s" data-id="%5$s">%1$s</figure>',
                            wp_get_attachment_image( $attachement_id, 'full', '', [ 'class' => 'gallery-img' ] ),
                            esc_url( wp_get_attachment_image_url( $attachement_id, 'full' ) ),
                            $prompt,
                            basename( get_attached_file( $attachement_id ) ),
                            esc_attr( $attachement_id )
                        );
                    }
                }
            }
        }

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'image',
            'ai' => 'sd',
            'count' => $this->sd_n
        ] );

        // add log
        MagicAI_Logs::instance()->add_log(
            'sd-image-completion',
            'Image Generated',
            wp_json_encode( [
                'prompt' => $prompt,
            ] ),
            1
        );

        wp_send_json( [
            'output' => $output,
        ] );

    }

    function remote_post( $type = 'text-to-image', $image = null, $opts ){

        if ( $type == 'text-to-image' ) {

            $url = "https://api.stability.ai/v1/generation/{$this->sd_model}/text-to-image";

            $headers = [
                "Content-Type: application/json",
                "Accept: application/json",
                "Authorization: Bearer " . $this->sd_key,
            ];

            $curl_info = [
                CURLOPT_URL => $url,
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_POST => true,
                CURLOPT_TIMEOUT => 300,
                CURLOPT_POSTFIELDS => json_encode($opts),
                CURLOPT_HTTPHEADER => $headers,
            ];


        } elseif ( $type === 'image-to-image' ) {

            $url = "https://api.stability.ai/v1/generation/{$this->sd_model}/image-to-image";

            $headers = [
                "Content-Type: multipart/form-data",
                'Accept: application/json',
                "Authorization: Bearer " . $this->sd_key,
            ];

            $data = [
                "init_image" =>  magicai_helper()->get_image_curlfile( $image ),
                "image_strength" => 0.35,
                "init_image_mode" => "IMAGE_STRENGTH",
                "text_prompts[0][text]" => $opts['text_prompts'][0]['text'],
                "cfg_scale" => 7,
                "samples" => $this->sd_n,
                "steps" => $opts['steps']
            ];

            $curl_info = [
                CURLOPT_URL => $url,
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_POST => true,
                CURLOPT_ENCODING => '',
                CURLOPT_MAXREDIRS => 10,
                CURLOPT_TIMEOUT => 300,
                CURLOPT_FOLLOWLOCATION => true,
                CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
                CURLOPT_POSTFIELDS => $data,
                CURLOPT_HTTPHEADER => $headers,
            ];

        } elseif ( $type === 'upscale' ) {
            
            $url = "https://api.stability.ai/v1/generation/esrgan-v1-x2plus/image-to-image/upscale";

            $headers = [
                "Content-Type: multipart/form-data",
                'Accept: application/json',
                "Authorization: Bearer " . $this->sd_key,
            ];

            $data = [
                "image" =>  magicai_helper()->get_image_curlfile( $image ),
            ];

            $curl_info = [
                CURLOPT_URL => $url,
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_POST => true,
                CURLOPT_TIMEOUT => 300,
                CURLOPT_POSTFIELDS => $data,
                CURLOPT_HTTPHEADER => $headers,
            ];
        }

        $curl = curl_init();

        curl_setopt_array($curl, $curl_info);

        $response = curl_exec($curl);
        $info = curl_getinfo($curl);

        curl_close($curl);

        $d = json_decode( $response );

        if ( $d->message ) {
            MagicAI_Logs::instance()->add_log(
                'sd-image-completion',
                'Completion Failed',
                wp_json_encode( [
                    'prompt' => $prompt,
                    'error_details' => sprintf( '%s: %s', $d->name, $d->message )
                ] ),
                1
            );
        
            wp_send_json( [
                'error' => true,
                'message' => sprintf( '%s: %s', $d->name, $d->message ),
            ] );
        }

        return $response;

    }


}
MagicAI_StableDiffusion::instance();
