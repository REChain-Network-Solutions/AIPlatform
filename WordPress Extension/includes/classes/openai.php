<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

use Orhanerday\OpenAi\OpenAi;
use Firebase\JWT\JWT;
use Firebase\JWT\Key;
use Smalot\PdfParser\Parser;

#[AllowDynamicProperties]
class MagicAI_Actions {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Actions The single instance of the class.
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
	 * @return MagicAI_Actions An instance of the class.
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
        $this->openai_key               = magicai_helper()->get_option( 'openai_key' );
        $this->openai_model             = magicai_helper()->get_option( 'openai_model', 'gpt-3.5-turbo-instruct' );
        $this->openai_max_tokens        = intval( magicai_helper()->get_option( 'openai_max_tokens', 800 ) );
        $this->openai_temperature       = floatval( magicai_helper()->get_option( 'openai_temperature', 0.75 ) );
        $this->openai_frequency_penalty = floatval( magicai_helper()->get_option( 'openai_frequency_penalty', 0 ) );
        $this->openai_presence_penalty  = floatval( magicai_helper()->get_option( 'openai_presence_penalty', 0.6 ) );
        $this->dalle_model              = magicai_helper()->get_option( 'dalle_model', 'dall-e-2' );
        $this->dalle_n                  = intval( magicai_helper()->get_option( 'dalle_n', 1 ) );
        $this->dalle_size               = magicai_helper()->get_option( 'dalle_size', '256x256' );

        $this->openai_tts_model         = magicai_helper()->get_option( 'openai_tts_model', 'tts-1' );

        $this->google_search_api        = magicai_helper()->get_option( 'google_search_api' );
        $this->google_search_cx         = magicai_helper()->get_option( 'google_search_cx' );

        $this->openai_model_post        = magicai_helper()->get_option('openai_model_post_generator') ? magicai_helper()->get_option('openai_model_post_generator') : $this->openai_model;
        $this->openai_model_custom      = magicai_helper()->get_option('openai_model_custom_generator') ? magicai_helper()->get_option('openai_model_custom_generator') : $this->openai_model;
        $this->openai_model_chat        = magicai_helper()->get_option('openai_model_chat') ? magicai_helper()->get_option('openai_model_chat') : 'gpt-3.5-turbo';
        $this->openai_model_chatbot     = magicai_helper()->get_option('openai_model_chatbot') ? magicai_helper()->get_option('openai_model_chatbot') : 'gpt-3.5-turbo';
        $this->openai_model_assistant   = magicai_helper()->get_option('openai_model_assistant') ? magicai_helper()->get_option('openai_model_assistant') : $this->openai_model;
        
        $this->unsplash_api_key         = magicai_helper()->get_option('unsplash_api_key');

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

        add_action( 'wp_ajax_magicai_block_editor', [ $this, 'openai_request' ] );
        add_action( 'wp_ajax_magicai_update_post_list', [ $this, 'update_post_list' ] );
        add_action( 'wp_ajax_magicai_generate_post', [ $this, 'generate_post' ] );
        add_action( 'wp_ajax_magicai_generate_product', [ $this, 'generate_product' ] );
        add_action( 'wp_ajax_magicai_generate_code', [ $this, 'generate_code' ] );
        add_action( 'wp_ajax_magicai_generate_image', [ $this, 'generate_image' ] );
        add_action( 'wp_ajax_magicai_transcribe_audio', [ $this, 'transcribe_audio' ] );
        add_action( 'wp_ajax_magicai_generate_yt_post', [ $this, 'generate_yt_post' ] );
        add_action( 'wp_ajax_magicai_generate_rss_post', [ $this, 'generate_rss_post' ] );

        // add_action( 'wp_ajax_magicai_web_search', [ $this, '_web_search' ] );
        
        add_action( 'wp_ajax_magicai_web_search', [ $this, 'web_search' ] );
        
        add_action( 'wp_ajax_magicai_openaitts', [ $this, 'generate_tts' ] );

        add_action( 'wp_ajax_magicai_create_fine_tune', [ $this, 'create_fine_tune' ] );
        add_action( 'wp_ajax_magicai_delete_fine_tune', [ $this, 'delete_fine_tune' ] );

        add_action( 'wp_ajax_magicai_pdf_parse', [ $this, 'parse_pdf_and_embedding' ] );
        add_action( 'wp_ajax_magicai_getMostSimilarText', [ $this, 'getMostSimilarText' ] );

        add_action( 'admin_enqueue_scripts', function() {

            $len = strlen($this->openai_key);

            $parts[] = substr($this->openai_key, 0, $l[] = rand(1, $len - 5));
            $parts[] = substr($this->openai_key, $l[0], $l[] = rand(1, $len - $l[0] - 3));
            $parts[] = substr($this->openai_key, array_sum($l));

            $apikeyPart1 = base64_encode($parts[0]);
            $apikeyPart2 = base64_encode($parts[1]);
            $apikeyPart3 = base64_encode($parts[2]);
            $apiUrl = base64_encode('https://api.openai.com/v1/chat/completions');

            wp_localize_script( 'magicai-admin-settings', 'magicai_js_options', array(
                'ajaxurl' => admin_url('admin-ajax.php'),
                'guest_id' => $apiUrl,
                'guest_event_id' => $apikeyPart1,
                'guest_look_id' => $apikeyPart2,
                'guest_product_id' => $apikeyPart3,
                'guest_status' => base64_encode(MagicAI_License::instance()->get_license_status()),
                'logo_url' => MAGICAI_URL . 'assets/img/logo.svg',
                'user_avatar' => get_avatar_url(get_current_user_id()),
                'model' => [
                    'post' => $this->openai_model_post,
                    'custom' => $this->openai_model_custom,
                    'chat' => $this->openai_model_chat,
                    'chatbot' => $this->openai_model_chatbot,
                    'assistant' => $this->openai_model_assistant,
                ],
                'modal' => [
                    'title' => [
                        'brand' => 'MagicAI',
                        'assistant' => 'MagicAI Assistant',
                        'add_fine_tune' => esc_html__( 'MagicAI - Add Fine Tune', 'magicai-wp' ),
                        'delete_fine_tune' => esc_html__( 'MagicAI - Delete Fine Tune', 'magicai-wp' ),
                        'chatbot_add_text' => esc_html__( 'MagicAI - ChatBot Add Text', 'magicai-wp' ),
                        'chatbot_add_qa' => esc_html__( 'MagicAI - ChatBot Add Q&A', 'magicai-wp' ),
                    ],
                    'content' => [
                        'assistant' => $this->assistant_modal_content(),
                        'prompt' => sprintf( 
                            '<div class="magicai-modal--prompt-wrapper">
                                <input class="magicai-modal--prompt" placeholder="%1$s" type="text" required />
                            </div>
                            <div class="magicai-prompt-examples">
                                <span>%2$s</span>
                                <span>%3$s</span>
                                <span>%4$s</span>
                            </div>',
                            esc_html__('Enter prompt...', 'magicai-wp'),
                            esc_html__('summarize', 'magicai-wp'),
                            esc_html__('translate to Spanish', 'magicai-wp'),
                            esc_html__('make more creative', 'magicai-wp'),
                        ),
                        'actions' => sprintf(
                            '<div class="magicai-modal--actions-wrapper">
                                <p class="loading-text">%1$s</p>
                                <div class="magicai-modal--actions">
                                    <div class="title">%2$s</div>
                                    <span data-prompt="title_grammar" data-prompt-type="title">%3$s</span>
                                    <span data-prompt="title_seo" data-prompt-type="title">%4$s</span>
                                    <span data-prompt="title_new" data-prompt-type="title">%5$s</span>
            
                                    <div class="title">%6$s</div>
                                    <span data-prompt="content_improve" data-prompt-type="content">%7$s</span>
                                    <span data-prompt="content_grammar" data-prompt-type="content">%3$s</span>
                                    <span data-prompt="content_tone" data-prompt-type="content">%8$s</span>
                                    <span data-prompt="content_reorganize" data-prompt-type="content">%9$s</span>
                                    <span data-prompt="content_translate" data-prompt-type="content">%10$s</span>
                                    <span data-prompt="content_expand" data-prompt-type="content">%11$s</span>
            
                                    <div class="title">%12$s</div>
                                    <span data-prompt="tag_generate_title" data-prompt-type="tag">%13$s</span>
                                    <span data-prompt="tag_generate_content" data-prompt-type="tag">%14$s</span>
                                    <span data-prompt="tag_add_more" data-prompt-type="tag">%15$s</span>
            
                                    <div class="title">%16$s</div>
                                    <span data-prompt="tools_duplicate" data-prompt-type="tools">%17$s</span>
                                </div>
                            </div>',
                            esc_html__('MagicAI is working on it...', 'magicai-wp'),
                            esc_html__('Title', 'magicai-wp'),
                            esc_html__('Fix Grammar', 'magicai-wp'),
                            esc_html__('Improve for SEO', 'magicai-wp'),
                            esc_html__('Create new one', 'magicai-wp'),
                            esc_html__('Content', 'magicai-wp'),
                            esc_html__('Improve', 'magicai-wp'),
                            esc_html__('Change Tone', 'magicai-wp'),
                            esc_html__('Reorganize for SEO', 'magicai-wp'),
                            esc_html__('Translate', 'magicai-wp'),
                            esc_html__('Expand', 'magicai-wp'),
                            esc_html__('Tags', 'magicai-wp'),
                            esc_html__('Generate tags about title', 'magicai-wp'),
                            esc_html__('Generate tags about content', 'magicai-wp'),
                            esc_html__('Add more tags', 'magicai-wp'),
                            esc_html__('Tools', 'magicai-wp'),
                            esc_html__('Duplicate Post', 'magicai-wp'),
                        ),
                        'translate' => '<div class="magicai-modal--prompt-wrapper">' . magicai_helper()->get_openai_languages_html() . '</div>',
                        'add_fine_tune' => sprintf( 
                            '<form class="magicai-form">
                                <div class="form-field">
                                    <label for="title">%1$s %2$s</label>
                                    <input type="text" id="title" name="title" placeholder="%3$s">
                                </div>

                                <div class="form-field">
                                    <label for="model">%4$s %5$s</label>
                                    <select name="model" id="model">
                                        <option value="gpt-3.5-turbo-1106">gpt-3.5-turbo-1106</option>
                                    </select>
                                </div>

                                <div class="form-field">
                                    <label for="purpose">%6$s %7$s</label>
                                    <select name="purpose" id="purpose">
                                        <option value="fine-tune">Fine Tune</option>
                                    </select>
                                </div>

                                <div class="form-field">
                                    <label for="file">%8$s %9$s</label>
                                    <input type="file" id="file" name="file" accept=".jsonl" style="padding:8px">
                                </div>


                            </form>',
                            magicai_helper()->label_help_tip( 'Enter a custom fine tune name.', false ),
                            esc_html__( 'Name', 'magicai-wp' ),
                            esc_html__( 'Enter name', 'magicai-wp' ),
                            magicai_helper()->label_help_tip( 'Select a model.', false ),
                            esc_html__( 'Model', 'magicai-wp' ),
                            magicai_helper()->label_help_tip( 'Select the purpose.', false ),
                            esc_html__( 'Purpose', 'magicai-wp' ),
                            magicai_helper()->label_help_tip( 'Select JSONL file', false ),
                            esc_html__( 'Select File (JSONL)', 'magicai-wp' ),
                        ),
                        'delete_fine_tune' => esc_html__( 'Your model and uploaded-file will be permanently deleted.', 'magicai-wp' ),
                        'chatbot_add_text' => sprintf(
                            '<div class="magicai-form">
                                <div class="form-field">
                                    <label for="title">%1$s</label>
                                    <input type="text" name="title" placeholder="%2$s">
                                </div>
                                <div class="form-field">
                                    <label for="content">%3$s</label>
                                    <textarea rows="6" name="content" placeholder="%4$s"></textarea>
                                </div>
                            </div>',
                            esc_html__( 'Title', 'magicai-wp' ),
                            esc_attr__( 'Title', 'magicai-wp' ),
                            esc_html__( 'Content', 'magicai-wp' ),
                            esc_attr__( 'Content', 'magicai-wp' ),
                        ),
                        'chatbot_add_qa' => sprintf(
                            '<div class="magicai-form">
                                <div class="form-field">
                                    <label for="q">%1$s</label>
                                    <input type="text" name="q" placeholder="%2$s">
                                </div>
                                <div class="form-field">
                                    <label for="a">%3$s</label>
                                    <textarea rows="6" name="a" placeholder="%4$s"></textarea>
                                </div>
                            </div>',
                            esc_html__( 'Question', 'magicai-wp' ),
                            esc_attr__( 'Question', 'magicai-wp' ),
                            esc_html__( 'Answer', 'magicai-wp' ),
                            esc_attr__( 'Answer', 'magicai-wp' ),
                        ),
                    ],
                ]
            ) );
        }, 90 );

        add_action( 'wp_enqueue_scripts', function() {

            $len = strlen($this->openai_key);

            $parts[] = substr($this->openai_key, 0, $l[] = rand(1, $len - 5));
            $parts[] = substr($this->openai_key, $l[0], $l[] = rand(1, $len - $l[0] - 3));
            $parts[] = substr($this->openai_key, array_sum($l));

            $apikeyPart1 = base64_encode($parts[0]);
            $apikeyPart2 = base64_encode($parts[1]);
            $apikeyPart3 = base64_encode($parts[2]);
            $apiUrl = base64_encode('https://api.openai.com/v1/chat/completions');

            wp_localize_script( 'magicai-admin-settings', 'magicai_js_options', array(
                'ajaxurl' => admin_url('admin-ajax.php'),
                'guest_id' => $apiUrl,
                'guest_event_id' => $apikeyPart1,
                'guest_look_id' => $apikeyPart2,
                'guest_product_id' => $apikeyPart3,
                'guest_status' => base64_encode(MagicAI_License::instance()->get_license_status()),
                'logo_url' => MAGICAI_URL . 'assets/img/logo.svg',
                'user_avatar' => get_avatar_url(get_current_user_id()),
                'model' => [
                    'post' => $this->openai_model_post,
                    'custom' => $this->openai_model_custom,
                    'chat' => $this->openai_model_chat,
                    'chatbot' => $this->openai_model_chatbot,
                    'assistant' => $this->openai_model_assistant,
                ],
            ) );
        }, 90 );

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

    /**
     * Check if the OpenAI model is a chat model.
     * 
     * @since 1.0.0
     *
     * @return bool Returns true if the model is a chat model, false otherwise.
     */
    public function is_chat_model( $model = null ) {

        if ( !empty( $model ) && $model == 'gpt-3.5-turbo-instruct' ) {
            return false;
        }

        if ( $this->openai_model == 'gpt-3.5-turbo-instruct' ) {
            return false;
        }

        return true;
        
    }

    /**
     * Get Prompts
     *
     * This function retrieves a list of predefined prompts for use with the MagicAI plugin.
     * It provides a set of prompts that can be used as input to request specific AI-generated content.
     *
     * @since 1.0.0
     *
     * @param string|null $prompt (Optional) The name of a specific prompt to retrieve.
     * If provided, returns the corresponding prompt text.
     * If not provided, returns an associative array of all available prompts.
     *
     * @return mixed|array|string|null Depending on the input, it returns a specific prompt, an array of prompts,
     * or null if the provided prompt does not exist.
     *
     * @access public
     */
    public function prompts( $prompt = null ) {

        $prompts = [
            // block editor
            'fix_grammar' => 'fix grammar',
            'summarize' => 'summarize',
            'expand' => 'expand content',
            'translate' => 'translate to TR',

            // Quick actions
            'title_grammar' => 'fix this blog post title grammar:',
            'title_seo' => 'improve blog post title for seo',
            'title_new' => 'create new blog post title',

            'content_improve' => 'improve blog post content',
            'content_grammar' => 'fix blog post content grammar',
            'content_tone' => 'change tone to funny',
            'content_reorganize' => 'reorganize blog post content for seo',
            'content_translate' => 'tranlate to turkish',
            'content_expand' => 'expand content',

            'tag_generate_title' => 'generate 5 tags about (return only comma separated):',
            'tag_generate_content' => 'generate 5 tags about (return only comma separated):',
            'tag_add_more' => 'add more post 5 tags about (return only comma separated):',
        ];

        if ( $prompt ) {
            if ( isset($prompts[$prompt]) ) {
                return $prompts[$prompt];
            }
            return $prompt;
        }

        return $prompts;

    }

    /**
     * OpenAI Request
     *
     * This function sends a request to the OpenAI API for text generation using provided input.
     * It combines a user-defined prompt with content and sends the request to generate AI-generated text.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function openai_request() {

        MagicAI_License::instance()->ajax_message();

        $open_ai = new OpenAi($this->openai_key);

        $prompt = sprintf( '%s %s', $this->prompts($_POST['data']['prompt']), $_POST['data']['content'] );
        $esc_attr = false;

        if ( isset( $_POST['data']['return_esc_attr'] ) && $_POST['data']['return_esc_attr'] == 'yes' ){
            $esc_attr = true;
        }

        if ( $this->is_chat_model() ) {
            $chat = $open_ai->chat([
                'model' => $this->openai_model_assistant,
                'messages' => [
                    [
                        "role" => "user",
                        "content" => $prompt,
                    ],
                ],
                'temperature' => $temperature,
                'max_tokens' => $this->openai_max_tokens,
                'frequency_penalty' => $this->openai_frequency_penalty,
                'presence_penalty' => $this->openai_presence_penalty,
             ]);

             $d = json_decode($chat);

            // check error
            if ( $d->error ) {
                MagicAI_Logs::instance()->add_log(
                    'completion',
                    'Completion Failed',
                    $d->error->message,
                );
    
                wp_send_json([
                    'error' => true,
                    'message' => $d->error->message
                ]);
            }
    
            $result = $d->choices[0]->message->content;
            $total_tokens = $d->usage->total_tokens;

        } else {
            $complete = $open_ai->completion([
                'model' => $this->openai_model_assistant,
                'prompt' => $prompt,
                'temperature' => $this->openai_temperature,
                'max_tokens' => $this->openai_max_tokens,
                'frequency_penalty' => $this->openai_frequency_penalty,
                'presence_penalty' => $this->openai_presence_penalty,
            ]);
    
            $d = json_decode($complete);

            // check error
            if ( $d->error ) {
                MagicAI_Logs::instance()->add_log(
                    'completion',
                    'Completion Failed',
                    $d->error->message,
                );
    
                wp_send_json([
                    'error' => true,
                    'message' => $d->error->message
                ]);
            }

            $result = $d->choices[0]->text;
            $total_tokens = $d->usage->total_tokens;
        }

        $log_data = [
            'prompt' => $prompt,
            'created' => $d->created,
            'id' => $d->id,
            'model' => $d->model,
            'model' => $d->object,
            'usage' => $d->usage,
        ];

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'text',
            'ai' => 'assistant',
            'count' => magicai_helper()->get_word_count( $result ),
        ] );

        // add log
        MagicAI_Logs::instance()->add_log(
            'completion',
            'Completion Worked',
            wp_json_encode( $log_data ),
            $total_tokens
        );

        if ( $esc_attr ) {
            $result = esc_attr($result);
        }

        wp_send_json( [
            'output' => trim( $result ),
            'd' => $d,
            'prompt' => $prompt
        ] );

    }

    /**
     * Update Post Data
     *
     * This function updates the selected post content, data coming from the AI.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function update_post_list() {

        MagicAI_License::instance()->ajax_message();

        $post_id = intval( $_POST['data']['post_id'] );
        $type = sanitize_text_field( $_POST['data']['type'] );

        $postarr = array(
            'ID' => $post_id,
        );

        if ( $type == 'title' ) {
            $content = sanitize_text_field( $_POST['data']['content'] );
            $postarr['post_title'] = $content;
        } elseif ( $type == 'content' ) {
            $content = wp_kses_post( $_POST['data']['content'] );
            $postarr['post_content'] = $content;
        } elseif ( $type == 'tag' ) {
            $content = sanitize_text_field( $_POST['data']['content'] );
            wp_set_post_tags( $post_id, $content, false );
        } else {
            $content = sanitize_text_field( $_POST['data']['content'] );
        }

        wp_update_post( $postarr );

        $log_data = [
            'post_ID' => $post_id,
            'type' => $type
        ];

        MagicAI_Logs::instance()->add_log(
            'post_quick_action',
            'Post Updated',
            wp_json_encode( $log_data ),
        );

        wp_send_json_success('Post Updated');

    }

    /**
     * OpenAI Request
     *
     * This function sends a request to the OpenAI API for text generation using provided input.
     * It combines a user-defined prompt with content and sends the request to generate AI-generated text.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function generate_post() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $title             = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : '';
        $tag               = !empty( $_POST['tag'] ) ? sanitize_text_field( $_POST['tag'] ) : '';
        $language          = !empty( $_POST['language'] ) ? sanitize_text_field( $_POST['language'] ) : 'en-US';
        $maximum_lenght    = !empty( $_POST['maximum_lenght'] ) ? intval( sanitize_text_field( $_POST['maximum_lenght'] ) ) : 400;
        $number_of_results = !empty( $_POST['number_of_results'] ) ? intval( sanitize_text_field( $_POST['number_of_results'] ) ) : 1;
        $temperature       = !empty( $_POST['temperature'] ) ? floatval( sanitize_text_field( $_POST['temperature'] ) ) : 0.75;
        $tone              = !empty( $_POST['tone'] ) ? sanitize_text_field( $_POST['tone'] ) : 'Professional';
        $image             = !empty( $_POST['image'] ) ? sanitize_text_field( $_POST['image'] ) : '';

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));

                $this->openai_key = !empty($atts->api_key) ? $atts->api_key : $this->openai_key;
                $this->openai_model_post = !empty($atts->model) ? $atts->model : $this->openai_model_post;
                $this->openai_temperature = !empty($atts->temperature) ? floatval($atts->temperature) : $this->openai_temperature;
                $this->openai_max_tokens = !empty($atts->max_tokens) ? intval($atts->max_tokens) : $this->openai_max_tokens;
                $this->openai_frequency_penalty = !empty($atts->frequency_penalty) ? floatval($atts->frequency_penalty) : $this->openai_frequency_penalty;
                $this->openai_presence_penalty = !empty($atts->openai_presence_penalty) ? floatval($atts->openai_presence_penalty) : $this->openai_presence_penalty;
            
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

        if ( ! empty( $tag ) ) {
            $tag_prefix = magicai_helper()->get_option('prompt_post_generator_tag_prefix');
            if ( !empty( $tag_prefix ) ) {
                $tag = str_replace( '[tag]', $tag, $tag_prefix );
            } else {
                $tag = " and related post tags {$tag}";
            }
        }

        // prepare the request
        $posts = $log_data = [];
        $total_tokens = 0;
        $open_ai = new OpenAi($this->openai_key);

        $final_prompt_title = "write one blog post title about {$title}. The tone of voice should {$tone} in {$language} language";
        $custom_prompt_title = magicai_helper()->get_option('prompt_post_generator_title');
        if ( !empty( $custom_prompt_title ) ) {
            $final_prompt_title = str_replace( array( '[title]', '[tone]', '[language]' ), array( $title, $tone, $language ), $custom_prompt_title );
        }

        $final_prompt_content = "write blog post with html tags(h2,p,strong,etc.) about {$title}$tag. The tone of voice should {$tone} in {$language} language. And maximum lenght $maximum_lenght";
        $custom_prompt_content = magicai_helper()->get_option('prompt_post_generator_content');
        if ( !empty( $custom_prompt_content ) ) {
            $final_prompt_content = str_replace( array( '[title]', '[tone]', '[language]', '[maximum_lenght]', '[tag]' ), array( $title, $tone, $language, $maximum_lenght, $tag ), $custom_prompt_content );
        }

        $final_prompt_tags = "write 5 tags about (return only comma separated): about {$title}$tag. The tone of voice should {$tone} in {$language} language.";
        $custom_prompt_tags = magicai_helper()->get_option('prompt_post_generator_tag');
        if ( !empty( $custom_prompt_tags ) ) {
            $final_prompt_tags = str_replace( array( '[title]', '[tone]', '[language]', '[tag]' ), array( $title, $tone, $language, $tag ), $custom_prompt_tags );
        }

        $request_query = [
            'title' => [
                'prompt' => $final_prompt_title,
            ],
            'content' => [
                'prompt' => $final_prompt_content,
            ],
            'tags' => [
                'prompt' => $final_prompt_tags,
            ],
        ];

        foreach( $request_query as $key => $request ) {

            if ( $this->is_chat_model( $this->openai_model_post ) ) {

                for ( $i = 0; $i < $number_of_results; $i++ ) {
                    $chat = $open_ai->chat([
                        'model' => $this->openai_model_post,
                        'messages' => [
                            [
                                "role" => "user",
                                "content" => $request['prompt'],
                            ],
                        ],
                        'temperature' => $temperature,
                        'max_tokens' => $this->openai_max_tokens,
                        'frequency_penalty' => $this->openai_frequency_penalty,
                        'presence_penalty' => $this->openai_presence_penalty,
                     ]);
    
                     $d = json_decode($chat);
        
                    // check error
                    if ( $d->error ) {
                        MagicAI_Logs::instance()->add_log(
                            'completion',
                            'Completion Failed',
                            $d->error->message,
                        );
            
                        wp_send_json([
                            'error' => true,
                            'message' => $d->error->message
                        ]);
                    }
            
                    $result = $d->choices[0]->message->content;
                    $total_tokens += $d->usage->total_tokens;
                    $posts[$i][$key] = trim($result);
                }

            } else {
                $complete = $open_ai->completion([
                    'model' => $this->openai_model_post,
                    'prompt' => $request['prompt'],
                    'temperature' => $temperature,
                    'max_tokens' => $this->openai_max_tokens,
                    'frequency_penalty' => $this->openai_frequency_penalty,
                    'presence_penalty' => $this->openai_presence_penalty,
                    'n' => $number_of_results,
                ]);

                $d = json_decode($complete);
    
                // check error
                if ( $d->error ) {
                    MagicAI_Logs::instance()->add_log(
                        'completion',
                        'Completion Failed',
                        $d->error->message,
                    );
        
                    wp_send_json([
                        'error' => true,
                        'message' => $d->error->message
                    ]);
                }
        
                $results = $d->choices;
                $total_tokens += $d->usage->total_tokens;

                foreach( $results as $i => $result ) {
                    $posts[$i][$key] = trim($result->text);
                }
            }
            
        }

        // return the HTML output
        $output = '<h3>🎉 Magic is here!</h3><p>You can review, edit and save the results as a post.</p>';
        $count_for_stats = 0;
        foreach ( $posts as $post_key => $post ) {
            $post_title = magicai_helper()->remove_quotes( $post['title'] );
            $images = $image == 'unsplash' ? $this->get_images_from_unsplash( $post['tags'] ) : array('url' => '', 'html' => '');
            $output .= sprintf(
                '<button class="magicai-accordion">
                    <div class="magicai-accordion-trigger"></div>
                    <div class="magicai-accordion-title">
                        %1$s
                    </div>
                    <div class="magicai-accordion-actions">
                        <div class="magicai-accordion-action" data-postid="%7$s" data-action="save-post">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M18 19H19V6.82843L17.1716 5H16V9H7V5H5V19H6V12H18V19ZM4 3H18L20.7071 5.70711C20.8946 5.89464 21 6.149 21 6.41421V20C21 20.5523 20.5523 21 20 21H4C3.44772 21 3 20.5523 3 20V4C3 3.44772 3.44772 3 4 3ZM8 14V19H16V14H8Z"></path></svg>
                            <span>Save as Draft</span>
                        </div>
                    </div>
                </button>
                <div class="magicai-accordion-panel">
                    <div class="form-field">
                        <label for="post_title_%7$s">%4$s</label>
                        <input type="text" name="post_title_%7$s" id="post_title_%7$s" value="%1$s">
                    </div>
                    <div class="form-field">
                        %8$s
                    </div>
                    <div class="form-field">
                        <label for="post_content_%7$s">%5$s</label>
                        <textarea name="post_content_%7$s" id="post_content_%7$s" cols="30" rows="10" aria-hidden="true">
                            %2$s
                        </textarea>
                    </div>
                    <div class="form-field">
                        <label for="post_tags_%7$s">%6$s</label>
                        <input type="text" name="post_tags_%7$s" id="post_tags_%7$s" value="%3$s">
                    </div>
                </div>',
                esc_attr( $post_title ),
                $image == 'unsplash' ? magicai_helper()->add_img_before_headings( $post['content'], $images['url'] ) : $post['content'],
                esc_attr( $post['tags'] ),
                'Post Title',
                'Post Content:',
                'Post Tags:',
                $post_key,
                $images['html']
            );

            // Save to documents
            magicai_helper()->save_to_documents([
                'title' => $post_title,
                'content' => $image == 'unsplash' ? magicai_helper()->add_img_before_headings( $post['content'], $images['url'] ) : $post['content'],
                'tags' => $post['tags'],
                'type' => 'post'
            ]);

            $log_data[] = $post_title;
            $count_for_stats += magicai_helper()->get_word_count( $post['content'] );
        }

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'text',
            'ai' => 'post_generator',
            'count' => $count_for_stats
        ] );

        // add log
        MagicAI_Logs::instance()->add_log(
            'completion',
            'Post Created',
            wp_json_encode( $log_data ),
            $total_tokens
        );

        wp_send_json( [
            'output' => $output,
            'n' => $number_of_results
        ] );

    }

    /**
     * OpenAI Request
     *
     * This function sends a request to the OpenAI API for text generation using provided input.
     * It combines a user-defined prompt with content and sends the request to generate AI-generated text.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function generate_product() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $title             = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : '';
        $language          = !empty( $_POST['language'] ) ? sanitize_text_field( $_POST['language'] ) : 'en-US';
        $maximum_lenght    = !empty( $_POST['maximum_lenght'] ) ? intval( sanitize_text_field( $_POST['maximum_lenght'] ) ) : 400;
        $number_of_results = !empty( $_POST['number_of_results'] ) ? intval( sanitize_text_field( $_POST['number_of_results'] ) ) : 1;
        $temperature       = !empty( $_POST['temperature'] ) ? floatval( sanitize_text_field( $_POST['temperature'] ) ) : 0.75;
        $tone              = !empty( $_POST['tone'] ) ? sanitize_text_field( $_POST['tone'] ) : 'Professional';

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));

                $this->openai_key = !empty($atts->api_key) ? $atts->api_key : $this->openai_key;
                $this->openai_model_post = !empty($atts->model) ? $atts->model : $this->openai_model_post;
                $this->openai_temperature = !empty($atts->temperature) ? floatval($atts->temperature) : $this->openai_temperature;
                $this->openai_max_tokens = !empty($atts->max_tokens) ? intval($atts->max_tokens) : $this->openai_max_tokens;
                $this->openai_frequency_penalty = !empty($atts->frequency_penalty) ? floatval($atts->frequency_penalty) : $this->openai_frequency_penalty;
                $this->openai_presence_penalty = !empty($atts->openai_presence_penalty) ? floatval($atts->openai_presence_penalty) : $this->openai_presence_penalty;
            
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

        // prepare the request
        $posts = $log_data = [];
        $total_tokens = 0;
        $open_ai = new OpenAi($this->openai_key);

        $final_prompt_title = "write one product title about {$title}. The tone of voice should {$tone} in {$language} language";
        $custom_prompt_title = magicai_helper()->get_option('prompt_product_generator_title');
        if ( !empty( $custom_prompt_title ) ) {
            $final_prompt_title = str_replace( array( '[title]', '[tone]', '[language]' ), array( $title, $tone, $language ), $custom_prompt_title );
        }

        $final_prompt_content = "write product content with html tags(h2,p,strong,etc.) about {$title}. The tone of voice should {$tone} in {$language} language. And maximum lenght $maximum_lenght";
        $custom_prompt_content = magicai_helper()->get_option('prompt_product_generator_content');
        if ( !empty( $custom_prompt_content ) ) {
            $final_prompt_content = str_replace( array( '[title]', '[tone]', '[language]', '[maximum_lenght]' ), array( $title, $tone, $language, $maximum_lenght ), $custom_prompt_content );
        }

        $final_prompt_tags = "write 5 tags about (return only comma separated): about {$title}. The tone of voice should {$tone} in {$language} language.";
        $custom_prompt_tags = magicai_helper()->get_option('prompt_product_generator_tag');
        if ( !empty( $custom_prompt_tags ) ) {
            $final_prompt_tags = str_replace( array( '[title]', '[tone]', '[language]' ), array( $title, $tone, $language ), $custom_prompt_tags );
        }

        $request_query = [
            'title' => [
                'prompt' => $final_prompt_title,
            ],
            'content' => [
                'prompt' => $final_prompt_content,
            ],
            'tags' => [
                'prompt' => $final_prompt_tags,
            ],
        ];

        foreach( $request_query as $key => $request ) {

            if ( $this->is_chat_model( $this->openai_model_post ) ) {
                for ( $i = 0; $i < $number_of_results; $i++ ) {
                    $chat = $open_ai->chat([
                        'model' => $this->openai_model_post,
                        'messages' => [
                            [
                                "role" => "user",
                                "content" => $request['prompt'],
                            ],
                        ],
                        'temperature' => $temperature,
                        'max_tokens' => $this->openai_max_tokens,
                        'frequency_penalty' => $this->openai_frequency_penalty,
                        'presence_penalty' => $this->openai_presence_penalty,
                     ]);
    
                     $d = json_decode($chat);
        
                    // check error
                    if ( $d->error ) {
                        MagicAI_Logs::instance()->add_log(
                            'completion',
                            'Completion Failed',
                            $d->error->message,
                        );
            
                        wp_send_json([
                            'error' => true,
                            'message' => $d->error->message
                        ]);
                    }
            
                    $result = $d->choices[0]->message->content;
                    $total_tokens += $d->usage->total_tokens;
                    $posts[$i][$key] = trim($result);
                }
            } else {
                $complete = $open_ai->completion([
                    'model' => $this->openai_model_post,
                    'prompt' => $request['prompt'],
                    'temperature' => $this->openai_temperature,
                    'max_tokens' => $this->openai_max_tokens,
                    'frequency_penalty' => $this->openai_frequency_penalty,
                    'presence_penalty' => $this->openai_presence_penalty,
                    'n' => $number_of_results,
                ]);

                $d = json_decode($complete);
    
                // check error
                if ( $d->error ) {
                    MagicAI_Logs::instance()->add_log(
                        'completion',
                        'Completion Failed',
                        $d->error->message,
                    );
        
                    wp_send_json([
                        'error' => true,
                        'message' => $d->error->message
                    ]);
                }
        
                $results = $d->choices;
                $total_tokens += $d->usage->total_tokens;

                foreach( $results as $i => $result ) {
                    $posts[$i][$key] = trim($result->text);
                }
            }
            
        }

        // return the HTML output
        $output = '<h3>🎉 Magic is here!</h3><p>You can review, edit and save the results as a post.</p>';
        $count_for_stats = 0;
        foreach ( $posts as $post_key => $post ) {
            $post_title = magicai_helper()->remove_quotes( $post['title'] );
            $output .= sprintf(
                '<button class="magicai-accordion">
                    <div class="magicai-accordion-trigger"></div>
                    <div class="magicai-accordion-title">
                        %1$s
                    </div>
                    <div class="magicai-accordion-actions">
                        <div class="magicai-accordion-action" data-postid="%7$s" data-action="save-post" data-type="product">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M18 19H19V6.82843L17.1716 5H16V9H7V5H5V19H6V12H18V19ZM4 3H18L20.7071 5.70711C20.8946 5.89464 21 6.149 21 6.41421V20C21 20.5523 20.5523 21 20 21H4C3.44772 21 3 20.5523 3 20V4C3 3.44772 3.44772 3 4 3ZM8 14V19H16V14H8Z"></path></svg>
                            <span>Save as Draft</span>
                        </div>
                    </div>
                </button>
                <div class="magicai-accordion-panel">
                    <div class="form-field">
                        <label for="post_title_%7$s">%4$s</label>
                        <input type="text" name="post_title_%7$s" id="post_title_%7$s" value="%1$s">
                    </div>
                    <div class="form-field">
                        <label for="product_content_%7$s">%5$s</label>
                        <textarea name="product_content_%7$s" id="product_content_%7$s" cols="30" rows="10" aria-hidden="true">
                            %2$s
                        </textarea>
                    </div>
                    <div class="form-field">
                        <label for="post_tags_%7$s">%6$s</label>
                        <input type="text" name="post_tags_%7$s" id="post_tags_%7$s" value="%3$s">
                    </div>
                </div>',
                esc_attr( $post_title ),
                $post['content'],
                esc_attr( $post['tags'] ),
                'Post Title',
                'Post Content:',
                'Post Tags:',
                $post_key
            );

            // Save to documents
            magicai_helper()->save_to_documents([
                'title' => $post_title,
                'content' => $post['content'],
                'tags' => $post['tags'],
                'type' => 'product'
            ]);

            $log_data[] = $post_title;
            $count_for_stats += magicai_helper()->get_word_count( $post['content'] );
        }

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'text',
            'ai' => 'product_generator',
            'count' => $count_for_stats
        ] );

        // add log
        MagicAI_Logs::instance()->add_log(
            'completion',
            'Post Created',
            wp_json_encode( $log_data ),
            $total_tokens
        );

        wp_send_json( [
            'output' => $output,
            'n' => $number_of_results
        ] );

    }

    /**
     * OpenAI Request
     *
     * This function sends a request to the OpenAI API for text generation using provided input.
     * It combines a user-defined prompt with content and sends the request to generate AI-generated text.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function generate_code() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $prompt = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : '';
        $code  = !empty( $_POST['code'] ) ? sanitize_text_field( $_POST['code'] ) : 'programmer';

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));

                $this->openai_key = !empty($atts->api_key) ? $atts->api_key : $this->openai_key;
                $this->openai_model = !empty($atts->model) ? $atts->model : $this->openai_model;
                $this->openai_temperature = !empty($atts->temperature) ? floatval($atts->temperature) : $this->openai_temperature;
                $this->openai_max_tokens = !empty($atts->max_tokens) ? intval($atts->max_tokens) : $this->openai_max_tokens;
                $this->openai_frequency_penalty = !empty($atts->frequency_penalty) ? floatval($atts->frequency_penalty) : $this->openai_frequency_penalty;
                $this->openai_presence_penalty = !empty($atts->openai_presence_penalty) ? floatval($atts->openai_presence_penalty) : $this->openai_presence_penalty;
            
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

        // prepare the request
        $open_ai = new OpenAi($this->openai_key);

        $final_prompt = "You are a {$code} expert. Write code about {$prompt}";
        $custom_prompt = magicai_helper()->get_option('prompt_code_generator');
        if ( !empty( $custom_prompt ) ) {
            $final_prompt = str_replace( array( '[code_language]', '[code_request]' ), array( $code, $prompt ), $custom_prompt );
        } 

        if ( $this->is_chat_model() ) {
            $chat = $open_ai->chat([
                'model' => $this->openai_model,
                'messages' => [
                    [
                        "role" => "user",
                        "content" => "You are a {$code} expert. Write code about {$prompt}",
                    ],
                ],
                'temperature' => $this->openai_temperature,
                'max_tokens' => $this->openai_max_tokens,
                'frequency_penalty' => $this->openai_frequency_penalty,
                'presence_penalty' => $this->openai_presence_penalty,
             ]);

             $d = json_decode($chat);

            // check error
            if ( $d->error ) {
                MagicAI_Logs::instance()->add_log(
                    'completion',
                    'Completion Failed',
                    $d->error->message,
                );
    
                wp_send_json([
                    'error' => true,
                    'message' => $d->error->message
                ]);
            }
    
            $result = $d->choices[0]->message->content;
            $total_tokens = $d->usage->total_tokens;

        } else {
            $complete = $open_ai->completion([
                'model' => $this->openai_model,
                'prompt' => $final_prompt,
                'temperature' => $this->openai_temperature,
                'max_tokens' => $this->openai_max_tokens,
                'frequency_penalty' => $this->openai_frequency_penalty,
                'presence_penalty' => $this->openai_presence_penalty,
            ]);
    
            $d = json_decode($complete);
    
            // check error
            if ( $d->error ) {
                MagicAI_Logs::instance()->add_log(
                    'completion',
                    'Completion Failed',
                    $d->error->message,
                );
    
                wp_send_json([
                    'error' => true,
                    'message' => $d->error->message
                ]);
            }
            
            $result = $d->choices[0]->text;
            $total_tokens = $d->usage->total_tokens;

        }

        $log_data = [
            'prompt' => $prompt,
            'created' => $d->created,
            'id' => $d->id,
            'model' => $d->model,
            'model' => $d->object,
            'usage' => $d->usage,
        ];

        // return the HTML output
        $output = '<h3>🎉 Magic is here!</h3><p>Here are your code snippets! 🖥️📝👨‍💻🤖💻🔍</p>';
        $output.= sprintf( 
            '<div class="theme-tomorrow"><pre><code class="line-numbers language-%s">%s</code></pre></div>',
            magicai_helper()->detect_programming_language($result),
            $result
        );

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'text',
            'ai' => 'code_generator',
            'count' => magicai_helper()->get_word_count( $result )
        ] );
       
        // add log
        MagicAI_Logs::instance()->add_log(
            'completion',
            'Code Generated',
            wp_json_encode( $log_data ),
            $total_tokens
        );

        $post_id = wp_insert_post( [
            'post_type' => 'magicai-documents',
            'post_status' => 'publish',
            'post_content' => $result,
            'post_title' => $prompt,
            'meta_input' => [ 
                '_magicai_doc_type' => 'code',
                '_magicai_userid' => get_current_user_id()
            ],
        ] );

        if ( is_wp_error( $post_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' =>  $post_id->get_error_message(),
            ] );
        }

        wp_send_json( [
            'output' => $output,
        ] );

    }

    /**
     * OpenAI Request
     *
     * This function sends a request to the OpenAI API for image generation using provided input.
     * It combines a user-defined prompt with content and sends the request to generate AI-generated image.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function generate_image() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $prompt = !empty( $_POST['prompt'] ) ? sanitize_text_field( $_POST['prompt'] ) : '';
        $image = !empty( $_POST['image'] ) ? intval( $_POST['image'] ) : '';
        $mask = !empty( $_POST['mask'] ) ? intval( $_POST['mask'] ) : '';
        $variation = !empty( $_POST['variation'] ) ? sanitize_text_field( $_POST['variation'] ) : 0;

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));
                $this->openai_key = !empty($atts->dalle_api_key) ? $atts->dalle_api_key : $this->openai_key;
                $this->dalle_n = !empty($atts->dalle_n) ? intval($atts->dalle_n) : $this->dalle_n;
                $this->dalle_size = !empty($atts->dalle_size) ? $atts->dalle_size : $this->dalle_size;
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

        // prepare the request
        $open_ai = new OpenAi($this->openai_key);

        if ( empty( $image ) ) {
            $complete = $open_ai->image([
                "model" => $this->dalle_model,
                "prompt" => $prompt,
                "n" => $this->dalle_n,
                "size" => $this->dalle_size,
                "response_format" => "url",
            ]);
        } else {
            if ( $variation ) {
                $complete = $open_ai->createImageVariation([
                    "image" => magicai_helper()->get_image_curlfile( $image ),
                    "n" => $this->dalle_n,
                    "size" => $this->dalle_size,
                ]);
            } else {
                $edit_image_arr = [
                    "image" => magicai_helper()->get_image_curlfile( $image ),
                    "prompt" => $prompt,
                    "n" => $this->dalle_n,
                    "size" => $this->dalle_size,
                ];
                if ( $mask ) { $edit_data['mask'] = magicai_helper()->get_image_curlfile( $mask ); }
                $complete = $open_ai->imageEdit( $edit_image_arr );
            }
        }

        $d = json_decode($complete);

        // check error
        if ( $d->error ) {
            MagicAI_Logs::instance()->add_log(
                'image-completion',
                'Completion Failed',
                $d->error->message,
            );

            wp_send_json([
                'error' => true,
                'message' => $d->error->message
            ]);
        }
        
        $output = '';
        foreach ( $d->data as $image ) {
            if ( magicai_helper()->get_option('storage', 'wp') == 's3' ) {
                $attachment_id = magicai_helper()->insert_image_to_s3( $image->url, $prompt );
                $output .= sprintf( 
                    '<figure class="gallery-item" data-src="%2$s" data-prompt="%3$s" data-name="%4$s" data-id="%5$s">%1$s</figure>',
                    '<img src="'.get_post_field( 'post_content', $attachment_id ).'" class="gallery-img">',
                    get_post_field( 'post_content', $attachment_id ),
                    $prompt,
                    basename( get_post_field( 'post_content', $attachment_id ) ),
                    esc_attr( $attachment_id )
                );
            } else {
                $attachment_id = magicai_helper()->insert_image_to_gallery( $image->url, $prompt, false, true );
                $output .= sprintf( 
                    '<figure class="gallery-item" data-src="%2$s" data-prompt="%3$s" data-name="%4$s" data-id="%5$s">%1$s</figure>',
                    wp_get_attachment_image( $attachment_id, 'full', '', [ 'class' => 'gallery-img' ] ),
                    esc_url( wp_get_attachment_image_url( $attachment_id, 'full' ) ),
                    $prompt,
                    basename( get_attached_file( $attachment_id ) ),
                    esc_attr( $attachment_id )
                );
            }
        }

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'image',
            'ai' => 'dall-e',
            'count' => $this->dalle_n
        ] );
       
        // add log
        MagicAI_Logs::instance()->add_log(
            'image-completion',
            'Image Generated',
            wp_json_encode( [
                'prompt' => $prompt,
                'created' => $d->created,
            ] ),
            1
        );

        wp_send_json( [
            'output' => $output,
        ] );

    }

    /**
     * OpenAI Request
     *
     * This function sends a request to the OpenAI API for transcribe audio using provided audio file.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function transcribe_audio() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $file = !empty( $_POST['file'] ) ? intval( $_POST['file'] ) : '';
        if ( empty( $file ) ) {
            wp_send_json([
                'error' => true,
                'message' => esc_html__( 'Media doesnt exist!', 'magicai-wp' )
            ]);
        }

        // prepare the request
        $open_ai = new OpenAi($this->openai_key);

        $c_file = curl_file_create(wp_get_attachment_url($file));

        $result = $open_ai->transcribe([
            "model" => "whisper-1",
            "file" => $c_file,
        ]);

        $d = json_decode($result);

        // check error
        if ( $d->error ) {
            MagicAI_Logs::instance()->add_log(
                'transcribe-audio',
                'Failed',
                $d->error->message,
            );

            wp_send_json([
                'error' => true,
                'message' => $d->error->message
            ]);
        }
        
        // add log
        MagicAI_Logs::instance()->add_log(
            'transcribe-audio',
            'Transcribe Generated',
            wp_json_encode( $c_file ),
            1
        );

        $post_id = wp_insert_post( [
            'post_type' => 'magicai-documents',
            'post_status' => 'publish',
            'post_content' => $d->text,
            'post_title' => basename( get_attached_file( $file ) ),
            'meta_input' => [ 
                '_magicai_doc_type' => 'transcribe',
                '_magicai_userid' => get_current_user_id()
            ],
        ] );

        if ( is_wp_error( $post_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' =>  $post_id->get_error_message(),
            ] );
        }

        $html = sprintf( 
            '<div class="doc-item">
                <div class="doc-item--title">%3$s %1$s</div>
                <div class="doc-item--content">%2$s</div>
            </div>',
            basename( get_attached_file( $file ) ),
            $d->text,
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12.4142 5H21C21.5523 5 22 5.44772 22 6V20C22 20.5523 21.5523 21 21 21H3C2.44772 21 2 20.5523 2 20V4C2 3.44772 2.44772 3 3 3H10.4142L12.4142 5ZM4 5V19H20V7H11.5858L9.58579 5H4ZM11 13.05V9H16V11H13V15.5C13 16.8807 11.8807 18 10.5 18C9.11929 18 8 16.8807 8 15.5C8 14.1193 9.11929 13 10.5 13C10.6712 13 10.8384 13.0172 11 13.05Z"></path></svg>'
        );

        wp_send_json( [
            'output' => $html,
        ] );

    }

    /**
     * Generate YouTube Post.
     *
     * This function processes a YouTube video URL, retrieves its transcript, and utilizes an AI model
     * to generate a blog post based on the video's transcript.
     *
     * @return void Returns the generated HTML output as JSON.
     * 
     * @since 1.0.0
     */
    function generate_yt_post() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $video_url = !empty( $_POST['url'] ) ? sanitize_url( $_POST['url'] ) : '';
        $language  = !empty( $_POST['language'] ) ? sanitize_text_field( $_POST['language'] ) : 'en-US';
        $action    = !empty( $_POST['yt_action'] ) ? sanitize_text_field( $_POST['yt_action'] ) : 'blog';

        if ( empty( $video_url ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'URL is not valid!', 'magicia-wp' ),
            ] );
        }

        // Parse the URL
        $parsedUrl = parse_url($video_url);

        // Check if it's a valid YouTube URL
        if (isset($parsedUrl['query'])) {
            // Parse the query string
            parse_str($parsedUrl['query'], $queryParameters);

            // Check if 'v' parameter exists in the query
            if (isset($queryParameters['v'])) {
                // Get the value of 'v'
                $video_id = $queryParameters['v'];
            }
        }

        $video_thumbnail = sprintf( 'https://img.youtube.com/vi/%s/maxresdefault.jpg', $video_id );
        $video_title = str_replace( ' - YouTube', '', explode('</title>', explode('<title>', file_get_contents($video_url))[1])[0] );
        
        $api_url = 'https://magicai-yt-video-post-api.vercel.app/api/transcript'; // Endpoint URL

        $request_args = array(
            'body' => json_encode(array(
                'video_url' => $video_url,
                'language' => 'en',
            )),
            'headers' => array(
                'Content-Type' => 'application/json',
            ),
            'timeout' => 300
        );

        $response = wp_remote_post($api_url, $request_args);

        if ( is_wp_error( $response ) ) {
            wp_send_json( [
                'error' => true,
                'message' => $response->get_error_message(),
            ] );
        } else {
            $response_code = wp_remote_retrieve_response_code( $response );
            $response_body = json_decode( wp_remote_retrieve_body( $response ) );

            $prompt = '';

            if ( $action == 'blog' ) {
                $prompt = "You are blog writer. Turn the given transcript text into a blog post in and translate to {$language} language. Group the content and create a subheading (witth HTML-h2) for each group. Content:";
                $custom_prompt = magicai_helper()->get_option('prompt_youtube_blog');
                if ( !empty( $custom_prompt ) ) {
                    $prompt = str_replace( '[language]', $language, $custom_prompt );
                }
            } elseif ( $action == 'short' ){
                $prompt = "You are transcript editor. Make sense of the given content and explain the main idea. Your result should be in {$language} language. Content:";
                $custom_prompt = magicai_helper()->get_option('prompt_youtube_short');
                if ( !empty( $custom_prompt ) ) {
                    $prompt = str_replace( '[language]', $language, $custom_prompt );
                }
            } elseif ( $action == 'list' ){
                $prompt = "You are transcript editor. Make sense of the given content and make a list main ideas. Your result should be in {$language} language. Content:";
                $custom_prompt = magicai_helper()->get_option('prompt_youtube_list');
                if ( !empty( $custom_prompt ) ) {
                    $prompt = str_replace( '[language]', $language, $custom_prompt );
                }
            } elseif ( $action == 'tldr' ){
                $prompt = "You are transcript editor. Make short TLDR. Your result should be in {$language} language. Content:";
                $custom_prompt = magicai_helper()->get_option('prompt_youtube_tldr');
                if ( !empty( $custom_prompt ) ) {
                    $prompt = str_replace( '[language]', $language, $custom_prompt );
                }
            } elseif ( $action == 'pros_cons' ){
                $prompt = "You are transcript editor. Make short pros and cons. Your result should be in {$language} language. Content:";
                $custom_prompt = magicai_helper()->get_option('prompt_youtube_pros_cons');
                if ( !empty( $custom_prompt ) ) {
                    $prompt = str_replace( '[language]', $language, $custom_prompt );
                }
            }

            if ( $response_code === 200 ) {
                $data = $response_body->result;
                foreach ( $data as $transcript ) {
                    $prompt .= $transcript->text .'<br>';
                }
            } else {
                wp_send_json( [
                    'error' => true,
                    'message' => $response_body->error,
                ] );
            }
        }

        $this->openai_model = 'gpt-3.5-turbo-16k';
        $this->openai_max_tokens = 8000;

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));

                $this->openai_key = !empty($atts->api_key) ? $atts->api_key : $this->openai_key;
                $this->openai_model = !empty($atts->model) ? $atts->model : $this->openai_model;
                $this->openai_temperature = !empty($atts->temperature) ? floatval($atts->temperature) : $this->openai_temperature;
                $this->openai_max_tokens = !empty($atts->max_tokens) ? intval($atts->max_tokens) : $this->openai_max_tokens;
                $this->openai_frequency_penalty = !empty($atts->frequency_penalty) ? floatval($atts->frequency_penalty) : $this->openai_frequency_penalty;
                $this->openai_presence_penalty = !empty($atts->openai_presence_penalty) ? floatval($atts->openai_presence_penalty) : $this->openai_presence_penalty;
            
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

        // prepare the request
        $open_ai = new OpenAi($this->openai_key);

        $complete = $open_ai->chat([
            'model' => $this->openai_model, // Minimum requirements: 16k-token
            'messages' => [
                [
                    "role" => "user",
                    "content" => $prompt
                ],
            ],
            'temperature' => $this->openai_temperature,
            'max_tokens' => $this->openai_max_tokens,
            'frequency_penalty' => $this->openai_frequency_penalty,
            'presence_penalty' => $this->openai_presence_penalty,
        ]);

        $d = json_decode($complete);

        // check error
        if ( $d->error ) {
            MagicAI_Logs::instance()->add_log(
                'completion',
                'Completion Failed',
                $d->error->message,
            );

            wp_send_json([
                'error' => true,
                'message' => $d->error->message
            ]);
        }
        
        $post_content = $d->choices[0]->message->content;
        $total_tokens = $d->usage->total_tokens;

        $log_data = [
            'created' => $d->created,
            'id' => $d->id,
            'model' => $d->model,
            'model' => $d->object,
            'usage' => $d->usage,
        ];

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'text',
            'ai' => 'youtube',
            'count' => magicai_helper()->get_word_count( $post_content )
        ] );

         // add log
         MagicAI_Logs::instance()->add_log(
            'youtube-video',
            'Post Created',
            wp_json_encode( $log_data ),
            $total_tokens
        );

        // return the HTML output
        $output = '<h3>🎉 Magic is here!</h3><p>You can review, edit and save the results as a post.</p>';

        if ( $action != 'blog' ) {

            $output .= sprintf(
                '<button class="magicai-accordion">
                    <div class="magicai-accordion-trigger active"></div>
                    <div class="magicai-accordion-title">%1$s</div>
                    <div class="magicai-accordion-actions">
                        <div class="magicai-accordion-action" data-postid="%2$s" data-action="copy-content">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6ZM7 11H13V13H7V11ZM7 15H13V17H7V15Z"></path></svg>                        <span>Copy</span>
                        </div>
                    </div>
                </button>
                <div class="magicai-accordion-panel" style="display:block">
                    <div class="form-field">
                        <label for="yt_post_content_%2$s">Result</label>
                        <textarea name="yt_post_content_%2$s" id="yt_post_content_%2$s" cols="30" rows="10" aria-hidden="true">%3$s</textarea>
                    </div>
                </div>',
                $video_title,
                esc_attr('1'),
                $post_content
            );

        } else {
            $output .= sprintf(
                '<button class="magicai-accordion">
                    <div class="magicai-accordion-trigger active"></div>
                    <div class="magicai-accordion-title">
                        %1$s
                    </div>
                    <div class="magicai-accordion-actions">
                        <div class="magicai-accordion-action" data-postid="%7$s" data-action="save-post" data-type="youtube">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M18 19H19V6.82843L17.1716 5H16V9H7V5H5V19H6V12H18V19ZM4 3H18L20.7071 5.70711C20.8946 5.89464 21 6.149 21 6.41421V20C21 20.5523 20.5523 21 20 21H4C3.44772 21 3 20.5523 3 20V4C3 3.44772 3.44772 3 4 3ZM8 14V19H16V14H8Z"></path></svg>
                            <span>Save as Draft</span>
                        </div>
                    </div>
                </button>
                <div class="magicai-accordion-panel">
                    <div class="form-field">
                        <img src="%3$s">
                        <input type="hidden" id="post_image_%7$s" value="%3$s">
                    </div>
                    <div class="form-field">
                        <label for="post_title_%7$s">%4$s</label>
                        <input type="text" name="post_title_%7$s" id="post_title_%7$s" value="%1$s">
                    </div>
                    <div class="form-field">
                        <label for="yt_post_content_%7$s">%5$s</label>
                        <textarea name="yt_post_content_%7$s" id="yt_post_content_%7$s" cols="30" rows="10" aria-hidden="true">
                            %2$s
                        </textarea>
                    </div>
                </div>',
                $video_title,
                $post_content,
                $video_thumbnail,
                'Post Title',
                'Post Content:',
                'Post Tags:',
                1,
            );
        }

        // Save to documents
        magicai_helper()->save_to_documents([
            'title' => $video_title,
            'content' => $post_content,
            'tags' => '',
            'type' => $action == 'blog' ? 'post' : 'text',
        ]);
 
         wp_send_json( [
             'output' => $output,
         ] );
    }

    /**
     * Generate RSS Post.
     *
     * This function processes a RSS Feed URL, retrieves its post data
     * to generate a blog post based on the related titles.
     *
     * @return void Returns the generated HTML output as JSON.
     * 
     * @since 1.0.0
     */
    function generate_rss_post() {

        // TODO: Add insert Image option for the next releases. 

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $url               = !empty( $_POST['url'] ) ? sanitize_url( $_POST['url'] ) : '';
        $title             = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : '';
        $tag               = !empty( $_POST['tag'] ) ? sanitize_text_field( $_POST['tag'] ) : '';
        $language          = !empty( $_POST['language'] ) ? sanitize_text_field( $_POST['language'] ) : 'en-US';
        $maximum_lenght    = !empty( $_POST['maximum_lenght'] ) ? intval( sanitize_text_field( $_POST['maximum_lenght'] ) ) : 400;
        $number_of_results = !empty( $_POST['number_of_results'] ) ? intval( sanitize_text_field( $_POST['number_of_results'] ) ) : 1;
        $temperature       = !empty( $_POST['temperature'] ) ? floatval( sanitize_text_field( $_POST['temperature'] ) ) : 0.75;
        $tone              = !empty( $_POST['tone'] ) ? sanitize_text_field( $_POST['tone'] ) : 'Professional';

        if ( !empty( $_POST['atts'] ) ) {
            try {
                $jwt = $_POST['atts'];
                $atts = JWT::decode($jwt, new Key( magicai_helper()->magicai_salt(), 'HS256'));

                $this->openai_key = !empty($atts->api_key) ? $atts->api_key : $this->openai_key;
                $this->openai_model_post = !empty($atts->model) ? $atts->model : $this->openai_model_post;
                $this->openai_temperature = !empty($atts->temperature) ? floatval($atts->temperature) : $this->openai_temperature;
                $this->openai_max_tokens = !empty($atts->max_tokens) ? intval($atts->max_tokens) : $this->openai_max_tokens;
                $this->openai_frequency_penalty = !empty($atts->frequency_penalty) ? floatval($atts->frequency_penalty) : $this->openai_frequency_penalty;
                $this->openai_presence_penalty = !empty($atts->openai_presence_penalty) ? floatval($atts->openai_presence_penalty) : $this->openai_presence_penalty;
            
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

        if ( empty( $url ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'URL is not valid!', 'magicia-wp' ),
            ] );
        }

        if ( empty( $title ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Title is not found!', 'magicia-wp' ),
            ] );
        }

        // prepare the request
        $posts = $log_data = [];
        $total_tokens = 0;
        $open_ai = new OpenAi($this->openai_key);

        $final_prompt_content = "write blog post with html tags(h2,p,strong,etc.) about {$title}. The tone of voice should {$tone} in {$language} language. And lenght should $maximum_lenght word";
        $custom_prompt_content = magicai_helper()->get_option('prompt_rss_generator_content');
        if ( !empty( $custom_prompt_content ) ) {
            $final_prompt_content = str_replace( array( '[title]', '[tone]', '[language]', '[maximum_lenght]' ), array( $title, $tone, $language, $maximum_lenght ), $custom_prompt_content );
        }

        $final_prompt_tags = "write 5 tags about (return only comma separated): about {$title}. The tone of voice should {$tone} in {$language} language.";
        $custom_prompt_tags = magicai_helper()->get_option('prompt_rss_generator_tags');
        if ( !empty( $custom_prompt_tags ) ) {
            $final_prompt_tags = str_replace( array( '[title]', '[tone]', '[language]' ), array( $title, $tone, $language ), $custom_prompt_tags );
        }

        $request_query = [
            // 'title' => [
            //     'prompt' => "write one blog post title about {$title}. The tone of voice should {$tone} in {$language} language",
            // ],
            'content' => [
                'prompt' => $final_prompt_content,
            ],
            'tags' => [
                'prompt' => $final_prompt_tags,
            ],
        ];

        foreach( $request_query as $key => $request ) {

            if ( $this->is_chat_model( $this->openai_model_post ) ) {

                for ( $i = 0; $i < $number_of_results; $i++ ) {
                    $chat = $open_ai->chat([
                        'model' => $this->openai_model_post,
                        'messages' => [
                            [
                                "role" => "user",
                                "content" => $request['prompt'],
                            ],
                        ],
                        'temperature' => $temperature,
                        'max_tokens' => $this->openai_max_tokens,
                        'frequency_penalty' => $this->openai_frequency_penalty,
                        'presence_penalty' => $this->openai_presence_penalty,
                     ]);
    
                     $d = json_decode($chat);
        
                    // check error
                    if ( $d->error ) {
                        MagicAI_Logs::instance()->add_log(
                            'completion',
                            'Completion Failed',
                            $d->error->message,
                        );
            
                        wp_send_json([
                            'error' => true,
                            'message' => $d->error->message
                        ]);
                    }
            
                    $result = $d->choices[0]->message->content;
                    $total_tokens += $d->usage->total_tokens;
                    $posts[$i][$key] = trim($result);
                }

            } else {
                $complete = $open_ai->completion([
                    'model' => $this->openai_model_post,
                    'prompt' => $request['prompt'],
                    'temperature' => $temperature,
                    'max_tokens' => $this->openai_max_tokens,
                    'frequency_penalty' => $this->openai_frequency_penalty,
                    'presence_penalty' => $this->openai_presence_penalty,
                    'n' => $number_of_results,
                ]);

                $d = json_decode($complete);
    
                // check error
                if ( $d->error ) {
                    MagicAI_Logs::instance()->add_log(
                        'completion',
                        'Completion Failed',
                        $d->error->message,
                    );
        
                    wp_send_json([
                        'error' => true,
                        'message' => $d->error->message
                    ]);
                }
        
                $results = $d->choices;
                $total_tokens += $d->usage->total_tokens;

                foreach( $results as $i => $result ) {
                    $posts[$i][$key] = trim($result->text);
                }
            }
            
        }

        // return the HTML output
        $output = '<h3>🎉 Magic is here!</h3><p>You can review, edit and save the results as a post.</p>';
        $count_for_stats = 0;
        foreach ( $posts as $post_key => $post ) {
            $post_title = magicai_helper()->remove_quotes( $title );
            $output .= sprintf(
                '<button class="magicai-accordion">
                    <div class="magicai-accordion-trigger"></div>
                    <div class="magicai-accordion-title">
                        %1$s
                    </div>
                    <div class="magicai-accordion-actions">
                        <div class="magicai-accordion-action" data-postid="%7$s" data-action="save-post">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M18 19H19V6.82843L17.1716 5H16V9H7V5H5V19H6V12H18V19ZM4 3H18L20.7071 5.70711C20.8946 5.89464 21 6.149 21 6.41421V20C21 20.5523 20.5523 21 20 21H4C3.44772 21 3 20.5523 3 20V4C3 3.44772 3.44772 3 4 3ZM8 14V19H16V14H8Z"></path></svg>
                            <span>Save as Draft</span>
                        </div>
                    </div>
                </button>
                <div class="magicai-accordion-panel">
                    <div class="form-field">
                        <label for="post_title_%7$s">%4$s</label>
                        <input type="text" name="post_title_%7$s" id="post_title_%7$s" value="%1$s">
                    </div>
                    <div class="form-field">
                        <label for="post_content_%7$s">%5$s</label>
                        <textarea name="post_content_%7$s" id="post_content_%7$s" cols="30" rows="10" aria-hidden="true">
                            %2$s
                        </textarea>
                    </div>
                    <div class="form-field">
                        <label for="post_tags_%7$s">%6$s</label>
                        <input type="text" name="post_tags_%7$s" id="post_tags_%7$s" value="%3$s">
                    </div>
                </div>',
                esc_attr( $post_title ),
                $post['content'],
                esc_attr( $post['tags'] ),
                'Post Title',
                'Post Content:',
                'Post Tags:',
                $post_key
            );

            // Save to documents
            magicai_helper()->save_to_documents([
                'title' => $post_title,
                'content' => $post['content'],
                'tags' => $post['tags'],
                'type' => 'post'
            ]);

            $log_data[] = $post_title;
            $count_for_stats += magicai_helper()->get_word_count( $post['content'] );
        }

        // add stats
        MagicAI_Stats::instance()->add_stats( [
            'type' => 'text',
            'ai' => 'youtube',
            'count' => $count_for_stats
        ] );

        // add log
        MagicAI_Logs::instance()->add_log(
            'completion',
            'Post Created',
            wp_json_encode( $log_data ),
            $total_tokens
        );

        wp_send_json( [
            'output' => $output,
            'n' => $number_of_results
        ] );
      
    }

    /**
     * [!!Deprecated!!] - v1.2
     * Search on web
     * alternative usage: $search_query = $this->improve_prompt_for_web_search( $search_query );
     */
    function _web_search() {

        $search_query = sanitize_text_field( $_POST['prompt'] );

        if ( empty( $search_query ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Prompt is empty', 'magicai-wp' )
            ] );
        }

        if ( empty( $this->google_search_api ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Custom Search JSON API: Missing API Key', 'magicai-wp' )
            ] );
        }

        if ( empty( $this->google_search_cx ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Custom Search JSON API: Missing Search Engine ID', 'magicai-wp' )
            ] );
        }

        $search_query = urlencode( $search_query );
        $url = "https://www.googleapis.com/customsearch/v1?key={$this->google_search_api}&cx={$this->google_search_cx}&q=$search_query";
        $response = file_get_contents($url);
        $results = json_decode($response, true);

        $web_results = '';

        // TODO: Add results count option
        if (isset($results['items'])) {
            $random_keys = array_rand($results['items'], 3);
        
            $i = 1;
            foreach ($random_keys as $key) {
                $item = $results['items'][$key];
                $web_results .= sprintf( '[%1$s] %2$s %3$s [%1$s-URL:%4$s]\n', ($i), $item['title'], $item['snippet'], $item['link'] );
                $i++;
            }
        }

        if ( ! $web_results ){
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Google Results not found. Try again or disable the web search feature!', 'magicai-wp' )
            ] );
        }

        wp_send_json( [
            'output' => sprintf(
                'Web search results:
                %s
                Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
                Query: %s',
                $web_results,
                $search_query
            )
        ] );
        

    }

    function improve_prompt_for_web_search( $prompt ) {

        $open_ai = new OpenAi($this->openai_key);

        $prompt = "Optimize this for web search best keyword. Write only the result, don't write anything else: $prompt";

        if ( $this->is_chat_model() ) {
            $chat = $open_ai->chat([
                'model' => $this->openai_model_post,
                'messages' => [
                    [
                        "role" => "user",
                        "content" => $prompt,
                    ],
                ],
                'temperature' => $temperature,
                'max_tokens' => $this->openai_max_tokens,
                'frequency_penalty' => $this->openai_frequency_penalty,
                'presence_penalty' => $this->openai_presence_penalty,
             ]);

             $d = json_decode($chat);

            // check error
            if ( $d->error ) {
                MagicAI_Logs::instance()->add_log(
                    'completion',
                    'Completion Failed',
                    $d->error->message,
                );
    
                wp_send_json([
                    'error' => true,
                    'message' => $d->error->message
                ]);
            }
    
            $result = $d->choices[0]->message->content;
            $total_tokens = $d->usage->total_tokens;

        } else {
            $complete = $open_ai->completion([
                'model' => $this->openai_model_post,
                'prompt' => $prompt,
                'temperature' => $this->openai_temperature,
                'max_tokens' => $this->openai_max_tokens,
                'frequency_penalty' => $this->openai_frequency_penalty,
                'presence_penalty' => $this->openai_presence_penalty,
            ]);
    
            $d = json_decode($complete);

            // check error
            if ( $d->error ) {
                MagicAI_Logs::instance()->add_log(
                    'completion',
                    'Completion Failed',
                    $d->error->message,
                );
    
                wp_send_json([
                    'error' => true,
                    'message' => $d->error->message
                ]);
            }

            $result = $d->choices[0]->text;
            $total_tokens = $d->usage->total_tokens;
        }

        return $result;

    }

    /**
     * Create Fine Tune
     *
     * This function is responsible for creating a fine-tune using OpenAI API.
     *
     * @since 1.0
     */
    function create_fine_tune() {

        // sanitize form data
        $title   = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : uniqid('model-');
        $purpose = !empty( $_POST['purpose'] ) ? sanitize_text_field( $_POST['purpose'] ) : 'fine-tune';
        $model   = !empty( $_POST['model'] ) ? sanitize_text_field( $_POST['model'] ) : 'gpt-3.5-turbo-1106';
        $file    = !empty( $_FILES['file'] ) ? $_FILES['file'] : array();

        $magicai_fine_tune = get_option( 'magicai_fine_tune', [] );

        $new_fine_tune = [];

        if ( empty( $title ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Title filed is empty!', 'magicai-wp' ),
            ] );
        }
        
        if ( empty( $file['name'] ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Select a JSONL File!', 'magicai-wp' ),
            ] );
        } else {
            $file_name = basename( $file['name'] );
            $tmp_file  = $file['tmp_name'];
            $file_type = $file['type'];
            $c_file    = curl_file_create($tmp_file, $file_type, $file_name);
        }

        // Openai
        $open_ai = new OpenAi($this->openai_key);

        // upload file
        $uploadFile = $open_ai->uploadFile([
            "purpose" => $purpose,
            "file" => $c_file,
        ]);
        $uploadFile_array = json_decode( $uploadFile, true );
        $uploadFile = json_decode( $uploadFile );

        if ( $uploadFile->error ) {
            MagicAI_Logs::instance()->add_log(
                'fine-tune',
                'Error',
                $uploadFile->error->message,
            );
            wp_send_json( [
                'error' => true,
                'message' => $uploadFile->error->message
            ] );
        }

        // create fine-tune
        $createFineTune = $open_ai->createFineTune([
                "model" => $model,
                "training_file" => $uploadFile->id,
        ]);
        $createFineTune = json_decode( $createFineTune );

        if ( $createFineTune->error ) {
            MagicAI_Logs::instance()->add_log(
                'fine-tune',
                'Error',
                $createFineTune->error->message,
            );
            wp_send_json( [
                'error' => true,
                'message' => $createFineTune->error->message
            ] );
        }

        // update data
        $magicai_fine_tune[$uploadFile->id] = [
            'title' => $title,
            'file' => $uploadFile_array,
        ];
        update_option( 'magicai_fine_tune', $magicai_fine_tune );

        MagicAI_Logs::instance()->add_log(
            'fine-tune',
            'New Model Created',
            sprintf( 'Title: %s, File ID: %s', $title, $uploadFile->id ),
        );

        wp_send_json( [
            'output' => sprintf( 
                '<tr>
                    <td>%1$s</td>
                    <td>%2$s</td>
                    <td>%3$s</td>
                    <td>%4$s</td>
                    <td>%5$s</td>
                    <td>%6$s %7$s</td>
                    <td><button class="delete" type="button">Delete</button></td>
                </tr>',
                $title,
                $uploadFile->id,
                $uploadFile->bytes,
                $model,
                $fine_tune->fine_tuned_model ?? '-',
                magicai_helper()->label_help_tip( 'Refresh the page after a few minutes', false ),
                $createFineTune->status,
            ),
        ] );

    }

    /**
     * Delete Fine Tune
     *
     * This function is responsible for deletimng a fine-tune using OpenAI API.
     *
     * @since 1.0
     */
    function delete_fine_tune() {

        // sanitize form data
        $file_id = !empty( $_POST['file_id'] ) ? sanitize_text_field( $_POST['file_id'] ) : '';
        $model   = !empty( $_POST['model'] ) ? sanitize_text_field( $_POST['model'] ) : '';

        $magicai_fine_tune = get_option( 'magicai_fine_tune', [] );
        
        if ( empty( $file_id ) && empty( $model ) ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Fine-tune not found!', 'magicai-wp' ),
            ] );
        }

        // Openai
        $open_ai = new OpenAi($this->openai_key);

        if ( $model ) {
            $deleteFineTune = $open_ai->deleteFineTune( $model );
            $deleteFineTune = json_decode( $deleteFineTune );

            if ( $deleteFineTune->error ) {
                MagicAI_Logs::instance()->add_log(
                    'fine-tune',
                    'Delete Error',
                    $deleteFineTune->error->message,
                );
                wp_send_json( [
                    'error' => true,
                    'message' => $deleteFineTune->error->message
                ] );
            }

            MagicAI_Logs::instance()->add_log(
                'fine-tune',
                'Fine Tune Deleted',
                $model,
            );

        }

        $deleteFile = $open_ai->deleteFile( $file_id );
        $deleteFile = json_decode( $deleteFile );

        if ( $deleteFile->error ) {
            MagicAI_Logs::instance()->add_log(
                'fine-tune',
                'Delete Error',
                $deleteFile->error->message,
            );
            wp_send_json( [
                'error' => true,
                'message' => $deleteFile->error->message
            ] );
        }

        if ( isset( $magicai_fine_tune[$file_id] ) ) {
            unset($magicai_fine_tune[$file_id]);
            update_option( 'magicai_fine_tune', $magicai_fine_tune );
        }

        MagicAI_Logs::instance()->add_log(
            'fine-tune',
            'File Deleted',
            $file_id,
        );

        wp_send_json([
            'deleted' => true,
            'message' => esc_html__( 'Fine-tune deleted!', 'magicai-wp' )
        ]);

    }

    /**
     * Get Fine Tunes Table
     *
     * Retrieves and displays a table of fine-tune data.
     *
     * @since 1.0
     */
    function get_fine_tune_table_row() {

        $open_ai = new OpenAi($this->openai_key);
        $magicai_fine_tune = get_option( 'magicai_fine_tune' );
        $html = '';

        $listFiles = json_decode( $open_ai->listFiles() );
        $listFineTunes = json_decode( $open_ai->listFineTunes() );

        if ( empty( $magicai_fine_tune ) ) {
            return printf( '<tr class="info"><td colspan="5">%s</td></tr>', esc_html__( 'There is no fine-tune data!', 'magicai-wp' ) );
        }

        foreach ( array_reverse($magicai_fine_tune) as $file_id => $values ) {
            foreach( $listFineTunes->data as $fine_tune ) {
                if ( $fine_tune->training_file == $file_id ){
                    $html .= sprintf( 
                        '<tr>
                            <td>%1$s</td>
                            <td>%2$s</td>
                            <td>%3$s</td>
                            <td>%4$s</td>
                            <td>%5$s</td>
                            <td>%6$s %7$s</td>
                            <td><button class="magicai-delete--fine-tune" type="button" data-file="%8$s" data-model="%9$s">%10$s</button></td>
                        </tr>',
                        $values['title'],
                        $file_id,
                        $values['file']['bytes'],
                        $fine_tune->model,
                        $fine_tune->fine_tuned_model ?? '-',
                        magicai_helper()->label_help_tip( $fine_tune->error->message ?? '', false ),
                        $fine_tune->status,
                        esc_attr($file_id),
                        esc_attr($fine_tune->fine_tuned_model),
                        esc_html__( 'Delete', 'magicai-wp' ),
                    );
                }
            }
        }

        echo $html;

    }

    /**
     * Get the list of fine-tuned models.
     *
     * @return array An array of fine-tuned models with their labels.
     * 
     * @since 1.0
     */
    function get_fine_tune_models() {

        $open_ai = new OpenAi($this->openai_key);
        $listFineTunes = json_decode( $open_ai->listFineTunes() );
        $magicai_fine_tune = get_option( 'magicai_fine_tune', [] );
        $arr = [
            '' => 'Use Default',
            'gpt-4' => 'gpt-4',
            'gpt-4-32k' => 'gpt-4-32k',
            'gpt-4-0125-preview' => 'gpt-4-0125-preview',
            'gpt-3.5-turbo' => 'gpt-3.5-turbo',
            'gpt-3.5-turbo-0125' => 'gpt-3.5-turbo-0125',
            'gpt-3.5-turbo-16k' => 'gpt-3.5-turbo-16k',
            'gpt-3.5-turbo-instruct' => 'gpt-3.5-turbo-instruct (Similar capabilities as text-davinci-003)',
        ];

        foreach ( $listFineTunes->data as $fine_tune ) {
            if ( $fine_tune->status == 'succeeded' && isset( $magicai_fine_tune[$fine_tune->training_file] ) ) {
               $arr[$fine_tune->fine_tuned_model] = $fine_tune->fine_tuned_model;
            }
        }

        return $arr;

    }

    /**
     * Generate the HTML content for the assistant modal.
     *
     * @return string HTML content for the assistant modal.
     * 
     * @since 1.0
     */
    function assistant_modal_content() {

        $prompts = magicai_helper()->get_option('assistant_prompts');
        $prompts = json_decode($prompts, true); 
        $prompts_html = '';

        foreach( $prompts as $prompt ) {
            $prompts_html .= sprintf(
                '<div class="magicai-assistant-prompt" data-prompt="%3$s">%1$s %2$s</div>',
                $prompt['icon'],
                $prompt['prompt'],
                esc_attr($prompt['prompt']),
            );
        }

        return sprintf(
            '<form id="magicai-assistant-form">
            <input type="text" class="magicai-assistant-form--prompt" placeholder="What would you like AI to do?" required>
                <button type="submit" class="magicai-assistant-form--submit"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M3.5 1.34558C3.58425 1.34558 3.66714 1.36687 3.74096 1.40747L22.2034 11.5618C22.4454 11.6949 22.5337 11.9989 22.4006 12.2409C22.3549 12.324 22.2865 12.3924 22.2034 12.4381L3.74096 22.5924C3.499 22.7255 3.19497 22.6372 3.06189 22.3953C3.02129 22.3214 3 22.2386 3 22.1543V1.84558C3 1.56944 3.22386 1.34558 3.5 1.34558ZM5 4.38249V10.9999H10V12.9999H5V19.6174L18.8499 11.9999L5 4.38249Z"></path></svg></button>
            </form>
            <div class="magicai-assistant-prompts">
                %s
                <div class="magicai-assistant-prompt" data-prompt="settings-assistant" data-href="%s">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path d="M2 18H9V20H2V18ZM2 11H11V13H2V11ZM2 4H22V6H2V4ZM20.674 13.0251L21.8301 12.634L22.8301 14.366L21.914 15.1711C21.9704 15.4386 22 15.7158 22 16C22 16.2842 21.9704 16.5614 21.914 16.8289L22.8301 17.634L21.8301 19.366L20.674 18.9749C20.2635 19.3441 19.7763 19.6295 19.2391 19.8044L19 21H17L16.7609 19.8044C16.2237 19.6295 15.7365 19.3441 15.326 18.9749L14.1699 19.366L13.1699 17.634L14.086 16.8289C14.0296 16.5614 14 16.2842 14 16C14 15.7158 14.0296 15.4386 14.086 15.1711L13.1699 14.366L14.1699 12.634L15.326 13.0251C15.7365 12.6559 16.2237 12.3705 16.7609 12.1956L17 11H19L19.2391 12.1956C19.7763 12.3705 20.2635 12.6559 20.674 13.0251ZM18 18C19.1046 18 20 17.1046 20 16C20 14.8954 19.1046 14 18 14C16.8954 14 16 14.8954 16 16C16 17.1046 16.8954 18 18 18Z"></path></svg>
                    Manage Prompts
                </div>
            </div>',
            $prompts_html,
            admin_url( 'admin.php?page=magicai-settings#tab-assistant' )
        );
    }

    /**
     * Get images from Unsplash based on provided tags.
     *
     * @param string $tags Comma-separated tags to search for images.
     * @return array An associative array containing image URLs and HTML content.
     * 
     * @since 1.0
     */
    function get_images_from_unsplash( $tags ) {

        $queries = explode( ',', sanitize_text_field( $tags ) );
        //shuffle($queries);
        $query = ltrim($queries[0]);

        if ( empty( $this->unsplash_api_key ) ) {

            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Unsplash API Key is missing! Go to the settings and add your API key', 'magicai-wp' ),
            ] );
        }

        $api_params = [
            'client_id' => $this->unsplash_api_key,
            'query' => $query,
            'per_page' => 9
        ];
    
        // https://unsplash.com/documentation
        $response = wp_remote_get( 
            add_query_arg( $api_params, "https://api.unsplash.com/search/photos" ),
            array( 'timeout' => 15 )
        );
    
        if ( ! is_wp_error( $response ) ) {
            $response_body = json_decode( wp_remote_retrieve_body( $response ), true );

            if ( $error = $response_body['errors'][0] ) {
                wp_send_json( [
                    'error' => true,
                    'message' => $error
                ] );
            }

            $image_urls = [];
            $response_body = $response_body['results'];
            $out = '<label>Select Featured Image</label><div class="generated-images-wrapper">';
                foreach ( $response_body as $key => $images ) { 
                    $image_urls[$key] = $images['urls']['full'];
                    if ( $key <= 3 ) {
                        $out .= '<div class="generated-images-option">';
                        $out .= sprintf( 
                                '<input type="radio" id="%s" name="generated-image" value="%s" %s>',
                                esc_attr( 'generated-image-' . $key ),
                                esc_url( $images['urls']['full'] ),
                                $key === 0 ? 'checked' : ''
                            );
                        $out .= sprintf( '<label for="%s"><img src="%s">%s %s</label></div>', esc_attr( 'generated-image-' . $key ), esc_url( $images['urls']['full'] ), esc_html__( 'Option', 'magicai-wp' ), ++$key  );
                    }
                } 
            $out .= '</div>';

            return [
                'url' => $image_urls,
                'html' => $out,
            ];
            
        } else {
            wp_send_json( [
                'error' => true,
                'message' => $response->get_error_message()
            ] );
        }
    
    }

    /**
     * Get audio files from OpenAI based on provided texts.
     *
     * @return file MP3 audio file.
     * 
     * @since 1.1
     */
    function generate_tts() {

        MagicAI_License::instance()->ajax_message();

        // sanitize form data
        $input = !empty( $_POST['input'] ) ? sanitize_text_field( $_POST['input'] ) : '';
        $voice = !empty( $_POST['voice'] ) ? sanitize_text_field( $_POST['voice'] ) : 'alloy';
        $voice_text = !empty( $_POST['voice_text'] ) ? sanitize_text_field( $_POST['voice_text'] ) : 'Alloy (Male)';
        $language = !empty( $_POST['language'] ) ? sanitize_text_field( $_POST['language'] ) : 'en-US';
       
        if ( empty( $input ) ) {
            wp_send_json([
                'error' => true,
                'message' => esc_html__( 'Speech doesnt exist!', 'magicai-wp' )
            ]);
        }

        // prepare the request
        $open_ai = new OpenAi($this->openai_key);

        $result = $open_ai->tts([
            "model" => $this->openai_tts_model,
            "input" => "$language: \"$input\"",
            "voice" => $voice,
        ]);

        $d = json_decode($result);

        // check error
        if ( $d->error ) {
            MagicAI_Logs::instance()->add_log(
                'voiceover',
                'Error',
                $d->error->message,
            );

            wp_send_json([
                'error' => true,
                'message' => $d->error->message
            ]);
        }
        // Save the image to the uploads directory
        $upload_dir = wp_upload_dir();
        $file_name = uniqid('openai-tts-').'.mp3';

        if (wp_mkdir_p($upload_dir['path'])) {
            $file = $upload_dir['path'] . '/' . $file_name;
        } else {
            $file = $upload_dir['basedir'] . '/' . $file_name;
        }

        file_put_contents($file, $result);

        $attachment = array(
            'post_title'     => $file_name,
            'post_mime_type' => 'audio/mpeg',
            'post_status'    => 'inherit',
        );

        $attachment_id = wp_insert_attachment($attachment, $file);

        if ( is_wp_error( $attachment_id ) ) {
            MagicAI_Logs::instance()->add_log(
                'voiceover',
                'Failed',
                $attachment_id->get_error_message(),
            );
            wp_send_json( [
                'error' => true,
                'message' => $attachment_id->get_error_message(),
            ] );
        }
            
        // Include necessary files
        require_once(ABSPATH . 'wp-admin/includes/image.php');

        // Generate attachment metadata
        $attachment_data = wp_generate_attachment_metadata($attachment_id, $file);

        // Update attachment metadata
        wp_update_attachment_metadata($attachment_id, $attachment_data);

        // Update attachment metadata with custom values
        update_post_meta($attachment_id, '_magicai_userid', get_current_user_id());
        update_post_meta($attachment_id, '_magicai_tts', 'openai');

        // Add file to documents.
        $post_content = sprintf( '<p>%s %s: %s</p>', magicai_helper()->country2flag($language), $voice_text, $input );
        $post_id = wp_insert_post( [
            'post_type' => 'magicai-documents',
            'post_status' => 'publish',
            'post_content' => $post_content,
            'post_title' => $file_name,
            'meta_input' => [ 
                '_magicai_doc_type' => 'voiceover',
                '_magicai_tts' => 'openai',
                '_magicai_attachment_id' => $attachment_id,
                '_magicai_userid' => get_current_user_id()
            ],
        ] );

        if ( is_wp_error( $post_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' =>  $post_id->get_error_message(),
            ] );
        }

        $output = sprintf( 
            '<div class="doc-item">
                <div class="doc-item--title">%3$s %1$s</div>
                <div class="doc-item--content">
                    %4$s
                    <div class="data-audio" data-audio="%2$s">
                        <div class="audio-preview"></div>
                    </div>
                </div>
            </div>',
            $file_name,
            wp_get_attachment_url($attachment_id),
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12.4142 5H21C21.5523 5 22 5.44772 22 6V20C22 20.5523 21.5523 21 21 21H3C2.44772 21 2 20.5523 2 20V4C2 3.44772 2.44772 3 3 3H10.4142L12.4142 5ZM4 5V19H20V7H11.5858L9.58579 5H4ZM11 13.05V9H16V11H13V15.5C13 16.8807 11.8807 18 10.5 18C9.11929 18 8 16.8807 8 15.5C8 14.1193 9.11929 13 10.5 13C10.6712 13 10.8384 13.0172 11 13.05Z"></path></svg>',
            $post_content,
        );

        wp_send_json( [
            'output' => $output
        ] );

    }

    /**
     * Custom Generator Function
     *
     * This function generates a response based on a prompt and, if applicable, web search results.
     *
     * @since 1.2
     *
     * @return string The generated response based on the provided prompt and web search results.
     */
    function web_search() {

        MagicAI_License::instance()->ajax_message();

        $prompt = !empty( $_POST['prompt'] ) ? sanitize_text_field( $_POST['prompt'] ) : '';
        $web_search = !empty( $_POST['web_search'] ) ? sanitize_text_field( $_POST['web_search'] ) : '';

       
        if ( $web_search ) {
            $search_results = magicai_helper()->get_serp_data($prompt);
            if ( empty( $search_results ) ) {
                wp_send_json([
                    'prompt' => $prompt
                ]);
            }
            $prompt =
            "Prompt: " . $prompt.
            '\n\nWeb search json results: '
            .$search_results.
            '\n\nInstructions: Based on the Prompt generate a proper response with help of Web search results(if the Web search results in the same context). Only if the prompt require links: (make curated list of links and descriptions using only the <a target="_blank">, write links with using <a target="_blank"> with mrgin Top of <a> tag is 5px and start order as number and write link first and then write description). Must not write links if its not necessary. Must not mention anything about the prompt text.';
        }

        wp_send_json([
            'prompt' => $prompt
        ]);

    }

    function parse_pdf_and_embedding() {

        $open_ai = new OpenAi($this->openai_key);

        $post_id = intval( $_POST['post_id'] );
        $attachment_id = intval( $_POST['attachment_id'] );

        $parser = new Parser;
        $text = $parser->parseFile(get_attached_file( $attachment_id ))->getText();
        
        $page = $text;
        if (!mb_check_encoding($text, 'UTF-8')) {
            $page = mb_convert_encoding($text, 'UTF-8', mb_detect_encoding($text));
        } else {
            $page = $text;
        }

        
        $countwords = strlen($page) / 500 + 1;
        $meta_index = 1;
        for ($i = 0; $i < $countwords; $i++) {
            if (500 * $i + 1000 > strlen($page)) {
                try {
                    $subtxt = substr($page, 500 * $i, strlen($page) - 500 * $i);
                    $subtxt = mb_convert_encoding($subtxt, 'UTF-8', 'UTF-8');
                    $subtxt = iconv('UTF-8', 'UTF-8//IGNORE', $subtxt);
                    $response = $open_ai->embeddings([
                        'model' => 'text-embedding-ada-002',
                        'input' => $subtxt,
                    ]);
                    $response = json_decode($response)->data[0]->embedding;
        
                    if (strlen(substr($page, 500 * $i, strlen($page) - 500 * $i)) > 10) {
                        $pdf_data = [
                            'content' => wp_kses_post(substr($page, 500 * $i, strlen($page) - 500 * $i)),
                            'vector' => json_encode($response),
                        ];
                        update_post_meta( $post_id, '_magicai_pdf_data_' . $meta_index, $pdf_data );
                        $meta_index++;
                    }
                } catch (Exception $e) {
                }
            } else {
                try {
                    $subtxt = substr($page, 500 * $i, 1000);
                    $subtxt = mb_convert_encoding($subtxt, 'UTF-8', 'UTF-8');
                    $subtxt = iconv('UTF-8', 'UTF-8//IGNORE', $subtxt);
                    $response = $open_ai->embeddings([
                        'model' => 'text-embedding-ada-002',
                        'input' => $subtxt
                    ]);

                    $response = json_decode($response)->data[0]->embedding;

                    if (strlen(substr($page, 500 * $i, 1000)) > 10) {

                        $pdf_data = [
                            'content' => wp_kses_post(substr($page, 500 * $i, 1000)),
                            'vector' => json_encode($response),
                        ];
                        update_post_meta( $post_id, '_magicai_pdf_data_' . $meta_index, $pdf_data );
                        $meta_index++;
                    }
                } catch (Exception $e) {
                }
            }

            update_post_meta( $post_id, '_magicai_pdf_data_count', ($meta_index - 1) );

            wp_send_json_success( 'ok' );

        }
    }

    function getMostSimilarText( ) {

        $post_id = intval( $_POST['post_id'] );
        $text = !empty( $_POST['prompt'] ) ? sanitize_text_field( $_POST['prompt'] ) : '';

        $vectors = get_post_meta( $post_id, '_magicai_pdf_data_count', true );
        if ( !$vectors ) {
            wp_send_json( [
                'extra_prompt' => null,
            ] );
        }

        $open_ai = new OpenAi($this->openai_key);
        $vector = $open_ai->embeddings([
            'model' => 'text-embedding-ada-002',
            'input' => $text
        ]);
        $vector = json_decode($vector)->data[0]->embedding;
        $similarVectors = [];
        for ($i=1; $i <= $vectors ; $i++) { 
            $v = get_post_meta( $post_id, '_magicai_pdf_data_' . $i, true );
            $cosineSimilarity = $this->calculateCosineSimilarity($vector, json_decode($v['vector']));
            $similarVectors[] = [
                'id' => wp_unique_id(),
                'content' => $v['content'],
                'similarity' => $cosineSimilarity
            ];
        }

        usort($similarVectors, function ($a, $b) {
            return $b['similarity'] <=> $a['similarity'];
        });

        $result = "";
        $resArr = array_slice($similarVectors, 0, 5);
        foreach ($resArr as $item) {
            $result = $result . $item['content'] . "\n\n\n";
        }
       
        wp_send_json( [
            'extra_prompt' => "'this pdf' means pdf content. Must not reference previous chats if user asking about pdf. Must reference pdf content if only user is asking about pdf. Else just response as an assistant shortly and professionaly without must not referencing pdf content. \n\n\n\n\nUser qusetion: $prompt \n\n\n\n\n PDF Content: \n $result"
        ] );
    }

    function calculateCosineSimilarity( $v1, $v2 ) {
        $dotProduct = 0;
        $v1Norm = 0;
        $v2Norm = 0;

        foreach ($v1 as $i => $value) {
            $dotProduct += $value * $v2[$i];
            $v1Norm += $value * $value;
            $v2Norm += $v2[$i] * $v2[$i];
        }

        $v1Norm = sqrt($v1Norm);
        $v2Norm = sqrt($v2Norm);

        return $dotProduct / ($v1Norm * $v2Norm);
    }

}
MagicAI_Actions::instance();