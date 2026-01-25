<?php

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * Class MagicAI_ChatBot
 *
 * A simple web crawler for extracting content and links from a given website.
 * @since 1.3
 */
class MagicAI_ChatBot {

    private $wpdb;

	private $table_name;
	const DB_VERSION = '1.0.0';

    /**
	 * Instance
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_ChatBot The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_ChatBot An instance of the class.
	 */
	public static function instance() {

		if ( is_null( self::$_instance ) ) {
			self::$_instance = new self();
		}
		return self::$_instance;
    }

    /**
     * Initializes the object by calling the init method.
     */
    public function __construct() {
        $this->db_init();
        $this->init();
    }

    /**
     * Initializes the object by setting up hooks.
     */
    public function init() {
		$this->hooks();
    }

    /**
     * Sets up WordPress hooks.
     */
    public function hooks() {

        add_action( 'wp_ajax_magicai_chatbot_train_url', [ $this, 'crawl_website' ] );
        add_action( 'wp_ajax_magicai_chatbot_train_single_url', [ $this, 'crawl_website_single_url' ] );
        add_action( 'wp_ajax_magicai_chatbot_train_with_url', [ $this, 'train_with_url' ] );
        add_action( 'wp_ajax_magicai_chatbot_delete_single_url', [ $this, 'delete_crawled_url' ] );

        add_action( 'wp_ajax_magicai_chatbot_train_with_pdf', [ $this, 'train_with_pdf' ] );
        add_action( 'wp_ajax_magicai_chatbot_delete_pdf', [ $this, 'delete_pdf' ] );

        add_action( 'wp_ajax_magicai_chatbot_train_with_text', [ $this, 'train_with_text' ] );
        add_action( 'wp_ajax_magicai_chatbot_delete_text', [ $this, 'delete_text' ] );
        add_action( 'wp_ajax_magicai_chatbot_get_train_text', [ $this, 'get_train_text' ] );
        
        add_action( 'wp_ajax_magicai_chatbot_train_with_qa', [ $this, 'train_with_qa' ] );
        add_action( 'wp_ajax_magicai_chatbot_delete_qa', [ $this, 'delete_qa' ] );
        add_action( 'wp_ajax_magicai_chatbot_get_train_qa', [ $this, 'get_train_qa' ] );

        add_action( 'wp_ajax_magicai_chatbot_getMostSimilarText', [ $this, 'getMostSimilarText' ] );
        add_action( 'wp_ajax_nopriv_magicai_chatbot_getMostSimilarText', [ $this, 'getMostSimilarText' ] );

        add_action( 'wp_ajax_magicai_chatbot_ban_ip', [ $this, 'ban_ip' ] );
       
    }

    /**
	 * Initialize DB
	 *
	 * Prepare the `wp_magicai_chatbot_data` database table.
	 */
    public function db_init() {
		global $wpdb;
		$this->wpdb = $wpdb;

		$this->table_name = $this->wpdb->prefix . 'magicai_chatbot_data';

		$query = $wpdb->prepare( 'SHOW TABLES LIKE %s', $wpdb->esc_like( $this->table_name ) );

		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		if ( $wpdb->get_var( $query ) !== $this->table_name ) {
			$this->create_table( $query );
			$this->add_indexes();
		}
	}

    /**
	 * Create Table
	 *
	 * Creates the `wp_magicai_chatbot_data` database table.
	 *
	 * @param string $query to that looks for the Events table in the DB. Used for checking if table was created.
	 */
	private function create_table( $query ) {
		require_once ABSPATH . 'wp-admin/includes/upgrade.php';

		$table_name = $this->table_name;
		$charset_collate = $this->wpdb->get_charset_collate();

		$sql = "CREATE TABLE `{$table_name}` (
			id bigint(20) unsigned auto_increment primary key,
			post_id int null,
			content text null,
			vector longtext null,
			type text null,
			type_value text null,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
		) {$charset_collate};";

		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		$this->wpdb->query( $sql );

		// Check if table was created successfully.
		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		if ( $this->wpdb->get_var( $query ) === $table_name ) {
			update_option( 'magicai_chatbot_data_db_version', self::DB_VERSION, false );
		}
	}

    /**
	 * Add Indexes
	 *
	 * Adds an index to the table for the creation date column.
	 */
	private function add_indexes() {
		// phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared
		$this->wpdb->query( 'ALTER TABLE ' . $this->table_name . '
    		ADD INDEX `created_at_index` (`created_at`)
		' );
	}

    /**
	 * Reset Table
	 *
	 * Empties the contents of the DB table.
	 */
	public static function reset_table() {
		global $wpdb;

		$table_name = $this->table_name;

		// Delete all content of the table.
		// phpcs:disable WordPress.DB.PreparedSQL.InterpolatedNotPrepared
		$wpdb->query( "TRUNCATE TABLE {$table_name}" );
	}

    /**
	 * Drop Table
	 *
	 * Delete the DB table.
	 */
	public static function drop_table() {
		global $wpdb;

		$table_name = $this->table_name;
		// Drop table
		// phpcs:disable WordPress.DB.PreparedSQL.InterpolatedNotPrepared
		$wpdb->query("DROP TABLE IF EXISTS {$table_name}");
	}

    /**
     * Delete Table Row
     *
     * Delete the DB table row.
     *
     * @param string $type       The type of the row to be deleted.
     * @param string $type_value The value of the type to be deleted.
     */
	public function delete_table_row( $type, $type_value ) {
        global $wpdb;

        $table_name = $this->table_name;

        $where = array(
            'type' => $type,
            'type_value' => $type_value
        );
        // Delete table rows
        // phpcs:disable WordPress.DB.PreparedSQL.InterpolatedNotPrepared
        $wpdb->delete( $table_name, $where );
	}

    /**
     * Callback function to handle AJAX request to crawl a website.
     *
     * @return void
     */
    function crawl_website() {

        $url = !empty( $_POST['url'] ) ? sanitize_url( $_POST['url'] ) : '';
        $url = wp_http_validate_url( $url );
        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';

        if ( !$url ) {
            wp_send_json( [
                'error' => true,
                'message' => __('URL is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        $magicai_train_url = get_post_meta( $post_id, '_magicai_train_url', true );
        if ( !is_array( $magicai_train_url ) ) {
            $magicai_train_url = array();
        }

        $crawler = new MagicAI_LinkCrawler($url);
        $crawler->crawl();
        $content = $crawler->getContents();
        $content = array_merge( $content, $magicai_train_url );

        update_post_meta( $post_id, '_magicai_train_url', $content);
        $trained_data = get_post_meta( $post_id, '_magicai_trained_url', true );

        wp_send_json( [
            'output' => $this->crawl_result_table( $content, $trained_data )
        ] );
    }

    /**
     * Callback function to handle AJAX request to crawl a website's single url.
     *
     * @return void
     */
    function crawl_website_single_url() {

        $url = !empty( $_POST['url'] ) ? sanitize_url( $_POST['url'] ) : '';
        $url = wp_http_validate_url( $url );
        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';

        if ( !$url ) {
            wp_send_json( [
                'error' => true,
                'message' => __('URL is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        $url = rtrim($url, "/") . "/";

        $magicai_train_url = get_post_meta( $post_id, '_magicai_train_url', true );
        if ( !is_array( $magicai_train_url ) ) {
            $magicai_train_url = array();
        }

        if ( isset( $magicai_train_url[$url] ) ) {
            wp_send_json( [
                'error' => true,
                'message' => __( 'URL already crawled!', 'magicai-wp' ),
            ] );
        }

        $crawler = new MagicAI_LinkCrawler($url);
        $crawler->crawl( $is_single = true );
        $content = $crawler->getContents();
        $content = array_merge( $content, $magicai_train_url );

        update_post_meta( $post_id, '_magicai_train_url', $content);
        $trained_data = get_post_meta( $post_id, '_magicai_trained_url', true );

        wp_send_json( [
            'output' => $this->crawl_result_table( $content, $trained_data )
        ] );
    }

    /**
     * Generates HTML markup for displaying crawled URLs and their word counts in a table format.
     *
     * This function takes an array of data containing URLs and their corresponding word counts
     * and generates HTML markup for displaying this information in a table. It also calculates
     * the total word count and displays it at the end of the table.
     *
     * @param array $data An associative array where keys are URLs and values are corresponding word counts.
     * @return string The HTML markup representing the crawled URLs and their word counts in a table.
     */
    function crawl_result_table( $data, $trained_data = array() ) {
        if ( !is_array( $trained_data ) ){
            $trained_data = array();
        }
        ob_start();
        ?>
        <div class="result">
            <h3><?php esc_html_e( 'Crawled URLs', 'magicai-wp' ); ?></h3>
            <table class="widefat striped" style="border-radius: var(--magicai-border-r);">
                <thead>
                    <th><?php esc_html_e( 'URL', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Status', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Word', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Action', 'magicai-wp' ); ?></th>
                </thead>
                <tbody>
                <?php
                    $total_word_count = 0;
                    foreach ( $data as $url => $value ) {
                        $word_count = magicai_helper()->get_word_count($value);
                        $total_word_count += $word_count;
                        printf(
                            '<tr>
                                <td>%1$s</td>
                                <td>%2$s %3$s</td>
                                <td>%4$s</td>
                                <td>
                                    <div class="actions">
                                        <div title="%5$s" class="magicai-chatbot--train-url--delete" data-url="%6$s" data-trained="%7$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM9 11H11V17H9V11ZM13 11H15V17H13V11ZM9 4V6H15V4H9Z"></path></svg></div>
                                    </div>
                                </td>
                            </tr>',
                            $url,
                            magicai_helper()->label_help_tip( in_array($url, $trained_data) ? 'This data was trained. Ready to use in the chatbot.' : 'Your action is needed. Please use the "Train" button to train the chatbot.', false ),
                            in_array($url, $trained_data) ? '<span class="trained">' . esc_html__('Trained') . '<span class="dashicons dashicons-yes"></span></span>' : esc_html__('Waiting the Action'),
                            $word_count,
                            esc_attr__( 'Delete', 'magicai-wp' ),
                            esc_attr( $url ),
                            in_array($url, $trained_data) ? 'yes' : 'no'
                        );
                    }
                ?>
                </tbody>
                <tfoot>
                    <td colspan="2"><?php printf( '%s: %s', __( 'Total URL', 'magicai-wp' ), count($data) ); ?></td>
                    <td colspan="2"><?php printf( '%s: %s', __( 'Total Word', 'magicai-wp' ), $total_word_count ); ?></td>
                </tfoot>
            </table>
            <br>
            <button type="button" class="magicai-btn magicai-chatbot--train-url--train"><?php esc_html_e( 'Train the ChatBot', 'magicai-wp' ); ?></button>
        </div>
        <?php 
        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;
    }

    /**
     * Deletes a crawled URL from the associated post's metadata.
     *
     * This function handles the deletion of a crawled URL from the metadata of the associated post.
     * It expects the URL and post ID to be sent via POST request. It performs validation on the URL
     * and post ID, and then proceeds to remove the URL from the metadata if it exists.
     *
     *
     * @return void This function doesn't return anything directly, but it sends a JSON response.
     */
    function delete_crawled_url() {
        $url = !empty( $_POST['url'] ) ? sanitize_url( $_POST['url'] ) : '';
        $url = wp_http_validate_url( $url );
        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $is_trained = !empty( $_POST['is_trained'] ) ? sanitize_text_field( $_POST['is_trained'] ) : 'no';

        if ( !$url ) {
            wp_send_json( [
                'error' => true,
                'message' => __('URL is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        $data = get_post_meta( $post_id, '_magicai_train_url', true );
        if ( isset( $data[$url] ) ){
            unset($data[$url]);
            update_post_meta( $post_id, '_magicai_train_url', $data );
        }

        if ( $is_trained == 'yes' ) {
            $this->delete_table_row( $type = 'url', $type_value = $url );
        }

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'Website Train Data Deleted',
            sprintf( 'Template Name: %s, Template ID: %s, URLs: %s', get_the_title( $post_id ), $post_id, $url )
        );

        wp_send_json( ['message' => __('Deleted', 'magicai-wp') ] );
    }

    /**
     * Trains the model with data from a given URL.
     * 
     * Retrieves the post ID from the POST request, then checks if it's valid.
     * If the post ID is not valid, sends a JSON response with an error message.
     * Otherwise, retrieves crawled and trained URLs data and parses and embeds the text data.
     *
     * @return void
     */
    function train_with_url() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        $crawled_urls = get_post_meta( $post_id, '_magicai_train_url', true );
        $trained_urls = get_post_meta( $post_id, '_magicai_trained_url', true );
        $this->parse_url_data_and_embedding( $post_id, $crawled_urls, $trained_urls );

    }

    /**
     * Parses URL data and embeds it into the model.
     * 
     * Parses the text data retrieved from crawled URLs, embeds it into the model,
     * and stores the embeddings along with other relevant data in the database.
     * 
     * @param int    $post_id      The ID of the post.
     * @param array  $crawled_urls An array of crawled URLs and their text data.
     * @param array  $trained_urls An array of URLs that have been trained.
     * 
     * @global object $wpdb WordPress database access object.
     *
     * @return void
     */
    function parse_url_data_and_embedding( $post_id, $crawled_urls = array(), $trained_urls = array() ) {

        global $wpdb;
        
        if ( !is_array( $trained_urls ) ) {
            $trained_urls = array();
        }

        $open_ai = new Orhanerday\OpenAi\OpenAi(magicai_helper()->get_option( 'openai_key' ));

        $text = '';
        $trained_urls_for_logs = [];
        foreach( $crawled_urls as $crawled_url => $crawled_text ) {
            if ( !in_array($crawled_url, $trained_urls) ) {
                $trained_urls[] = $crawled_url;
                $trained_urls_for_logs[] = $crawled_url;
                $text = $crawled_text;
        
                $page = $text;
                if (!mb_check_encoding($text, 'UTF-8')) {
                    $page = mb_convert_encoding($text, 'UTF-8', mb_detect_encoding($text));
                } else {
                    $page = $text;
                }
        
                $countwords = strlen($page) / 500 + 1;
                $train_array = [];
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
                                $wpdb->insert(
                                    $this->table_name,
                                    array(
                                        'content' => wp_kses_post(substr($page, 500 * $i, strlen($page) - 500 * $i)),
                                        'vector' => json_encode($response),
                                        'post_id' => $post_id,
                                        'type' => 'url',
                                        'type_value' => esc_url_raw( $crawled_url ),
                                    )
                                );
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
        
                                $wpdb->insert(
                                    $this->table_name,
                                    array(
                                        'content' => wp_kses_post(substr($page, 500 * $i, 1000)),
                                        'vector' => json_encode($response),
                                        'post_id' => $post_id,
                                        'type' => 'url',
                                        'type_value' => esc_url_raw( $crawled_url ),
                                    )
                                );
                                $meta_index++;
                            }
                        } catch (Exception $e) {
                        }
                    }
        
                }

            }
        }

        update_post_meta( $post_id, '_magicai_trained_url', $trained_urls );

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'Website Train Data Added',
            sprintf( 'Template Name: %s, Template ID: %s, URLs: %s', get_the_title( $post_id ), $post_id, join(", ", $trained_urls_for_logs) )
        );

        wp_send_json( [
            'output' => $this->crawl_result_table( get_post_meta( $post_id, '_magicai_train_url', true ), get_post_meta( $post_id, '_magicai_trained_url', true ) ),
            'message' => __( 'Tained Successful!', 'magicai-wp' )
        ] );

    }

    /**
     * Trains with a PDF file.
     *
     * This function handles the training process with a PDF file.
     * It retrieves necessary data from the $_POST superglobal, including the post ID and attachment ID.
     * It checks for the validity of the data and existence of the PDF file. Then,
     * it proceeds to parse the PDF data and embed it into vectors using OpenAI's text-embedding model.
     * The resulting data is stored in the database.
     *
     * @return void
     */
    function train_with_pdf() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $attachment_id = !empty( $_POST['attachment_id'] ) ? intval( $_POST['attachment_id'] ) : '';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$attachment_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('PDF File does not exists!', 'magicai-wp')
            ] );
        }

        $trained_pdf = get_post_meta( $post_id, '_magicai_trained_pdf', true );
        if ( isset( $trained_pdf[$attachment_id] ) ){
            wp_send_json( [
                'error' => true,
                'message' => __('This file is already trained before!', 'magicai-wp')
            ] );
        }
        $this->parse_pdf_data_and_embedding( $post_id, $attachment_id, $trained_pdf );

    }

    /**
     * Parses PDF data and embeds it into vectors.
     *
     * This function parses the content of a PDF file and embeds it into vectors using OpenAI's text-embedding model.
     * It iterates over the PDF content, breaks it into manageable chunks, and embeds each chunk into vectors.
     * The resulting data is stored in the database and updates post meta accordingly.
     *
     * @param int    $post_id        The ID of the post associated with the PDF file.
     * @param int    $attachment_id  The ID of the PDF file attachment.
     * @param array  $trained_pdf    Array containing information about trained PDF files.
     * 
     * @return void
     */
    function parse_pdf_data_and_embedding( $post_id, $attachment_id, $trained_pdf = array() ) {

        global $wpdb;
        
        if ( !is_array( $trained_pdf ) ) {
            $trained_pdf = array();
        }

        $open_ai = new Orhanerday\OpenAi\OpenAi(magicai_helper()->get_option( 'openai_key' ));
        $parser = new Smalot\PdfParser\Parser;
    

        if ( !in_array($attachment_id, $trained_pdf) ) {

            $text = $parser->parseFile(get_attached_file( $attachment_id ))->getText();
    
            $page = $text;
            if (!mb_check_encoding($text, 'UTF-8')) {
                $page = mb_convert_encoding($text, 'UTF-8', mb_detect_encoding($text));
            } else {
                $page = $text;
            }
    
            $countwords = strlen($page) / 500 + 1;
            $trained_pdf[$attachment_id] = [
                'filename' => basename( get_attached_file( $attachment_id ) ),
                'word_count' => magicai_helper()->get_word_count($text),
            ];
            
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
                            $wpdb->insert(
                                $this->table_name,
                                array(
                                    'content' => wp_kses_post(substr($page, 500 * $i, strlen($page) - 500 * $i)),
                                    'vector' => json_encode($response),
                                    'post_id' => $post_id,
                                    'type' => 'pdf',
                                    'type_value' => $attachment_id,
                                )
                            );
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
    
                            $wpdb->insert(
                                $this->table_name,
                                array(
                                    'content' => wp_kses_post(substr($page, 500 * $i, 1000)),
                                    'vector' => json_encode($response),
                                    'post_id' => $post_id,
                                    'type' => 'pdf',
                                    'type_value' => $attachment_id,
                                )
                            );
                        }
                    } catch (Exception $e) {
                    }
                }
    
            }

        }

        update_post_meta( $post_id, '_magicai_trained_pdf', $trained_pdf );

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'PDF Train Data Added',
            sprintf( 'Template Name: %s, Template ID: %s, File Name: %s', get_the_title( $post_id ), $post_id, $trained_pdf[$attachment_id]['filename'] )
        );

        wp_send_json( [
            'output' => $this->pdf_result_table( 
                get_post_meta( $post_id, '_magicai_trained_pdf', true )
            ),
            'message' => __( 'Tained Successful!', 'magicai-wp' )
        ] );

    }

    /**
     * Generates a HTML table displaying trained PDF files.
     *
     * This function generates a HTML table displaying information about trained PDF files.
     * It retrieves data from the database and formats it into a table. Each row represents a trained PDF file,
     * including its name, status, word count, and an option to delete it.
     *
     * @param array  $data  Array containing information about trained PDF files.
     * 
     * @return string HTML representation of the trained PDF files table.
     */
    function pdf_result_table( $data = array() ) {
        ob_start();
        ?>
        <div class="result">
            <h3><?php esc_html_e( 'Trained PDF Files', 'magicai-wp' ); ?></h3>
            <table class="widefat striped" style="border-radius: var(--magicai-border-r);">
                <thead>
                    <th><?php esc_html_e( 'File', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Status', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Word', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Action', 'magicai-wp' ); ?></th>
                </thead>
                <tbody>
                <?php
                    $total_word_count = 0;
                    foreach ( $data as $attachment_id => $pdf ) {
                        $word_count = $pdf['word_count'];
                        $total_word_count += $word_count;
                        printf(
                            '<tr>
                                <td>%1$s</td>
                                <td>%2$s %3$s</td>
                                <td>%4$s</td>
                                <td>
                                    <div class="actions">
                                        <div title="%5$s" class="magicai-chatbot--train-pdf--delete" data-id="%6$s" data-trained="%7$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM9 11H11V17H9V11ZM13 11H15V17H13V11ZM9 4V6H15V4H9Z"></path></svg></div>
                                    </div>
                                </td>
                            </tr>',
                            $pdf['filename'],
                            magicai_helper()->label_help_tip( 'This data was trained. Ready to use in the chatbot.', false ),
                            '<span class="trained">' . esc_html__('Trained') . '<span class="dashicons dashicons-yes"></span></span>',
                            $pdf['word_count'],
                            esc_attr__( 'Delete', 'magicai-wp' ),
                            esc_attr( $attachment_id ),
                            'yes'
                        );
                    }
                ?>
                </tbody>
                <tfoot>
                    <td colspan="2"><?php printf( '%s: %s', __( 'Total PDF File', 'magicai-wp' ), count($data) ); ?></td>
                    <td colspan="2"><?php printf( '%s: %s', __( 'Total Word', 'magicai-wp' ), $total_word_count ); ?></td>
                </tfoot>
            </table>
        </div>
        <?php 
        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;
    }

    /**
     * Deletes a trained PDF data.
     *
     * This function deletes a trained PDF data from the database.
     * It receives the post ID and attachment ID of the PDF data to be deleted.
     * Optionally, it can delete associated data from the database table. Upon successful deletion,
     * it sends a JSON response indicating success.
     *
     * @return void
     */
    function delete_pdf() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $attachment_id = !empty( $_POST['attachment_id'] ) ? intval( $_POST['attachment_id'] ) : '';
        $is_trained = !empty( $_POST['is_trained'] ) ? sanitize_text_field( $_POST['is_trained'] ) : 'no';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$attachment_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('PDF File does not exists!', 'magicai-wp')
            ] );
        }

        $data = get_post_meta( $post_id, '_magicai_trained_pdf', true );
        if ( isset( $data[$attachment_id] ) ){
            MagicAI_Logs::instance()->add_log(
                'chatbot-train',
                'PDF Train Data Deleted',
                sprintf( 'Template Name: %s, Template ID: %s, File Name: %s', get_the_title( $post_id ), $post_id, $data[$attachment_id]['filename'] )
            );
            unset($data[$attachment_id]);
            update_post_meta( $post_id, '_magicai_trained_pdf', $data );
        }

        if ( $is_trained == 'yes' ) {
            $this->delete_table_row( $type = 'pdf', $type_value = $attachment_id );
        }

        wp_send_json( ['message' => __('Deleted', 'magicai-wp') ] );
    }

    /**
     * Train with text data.
     * 
     * This function handles the training process with text data provided via POST request.
     * It sanitizes and validates the input data and then parses the text data and embeds it using OpenAI.
     *
     * @return void
     */
    function train_with_text() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $text_id = !empty( $_POST['text_id'] ) ? sanitize_text_field( $_POST['text_id'] ) : '';
        $title = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : '';
        $content = !empty( $_POST['content'] ) ? sanitize_textarea_field( $_POST['content'] ) : '';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$title ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Title does not exists!', 'magicai-wp')
            ] );
        }

        if ( !$content ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Content does not exists!', 'magicai-wp')
            ] );
        }

        if ( empty( $text_id ) ) {
            $text_id = uniqid();
        }

        $trained_text = get_post_meta( $post_id, '_magicai_trained_text', true );
        $this->parse_text_data_and_embedding( $post_id, $text_id, $title, $content, $trained_text );

    }

    /**
     * Parse text data and embed it.
     * 
     * This function parses the provided text data, splits it into smaller chunks, and embeds each chunk using OpenAI.
     * It then stores the embedded data in the database.
     *
     * @param int    $post_id      The ID of the post associated with the text data.
     * @param string $text_id      The ID of the text data.
     * @param string $title        The title of the text data.
     * @param string $content      The content of the text data.
     * @param array  $trained_text An array containing previously trained text data.
     * 
     * @return void
     */
    function parse_text_data_and_embedding( $post_id, $text_id, $title, $content, $trained_text = array() ) {

        global $wpdb;
        
        if ( !is_array( $trained_text ) ) {
            $trained_text = array();
        }

        $open_ai = new Orhanerday\OpenAi\OpenAi(magicai_helper()->get_option( 'openai_key' ));

        // if data exists before, delete from db
        if ( isset($trained_text[$text_id]) ) {
            $this->delete_table_row( $type = 'text', $type_value = $text_id );
        }
    
        $text = $content;
        $page = $text;

        if (!mb_check_encoding($text, 'UTF-8')) {
            $page = mb_convert_encoding($text, 'UTF-8', mb_detect_encoding($text));
        } else {
            $page = $text;
        }

        $countwords = strlen($page) / 500 + 1;
        $trained_text[$text_id] = [
            'title' => $title,
            'content' => $content,
            'word_count' => magicai_helper()->get_word_count($text),
        ];
        
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
                        $wpdb->insert(
                            $this->table_name,
                            array(
                                'content' => wp_kses_post(substr($page, 500 * $i, strlen($page) - 500 * $i)),
                                'vector' => json_encode($response),
                                'post_id' => $post_id,
                                'type' => 'text',
                                'type_value' => $text_id,
                            )
                        );
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
                        $wpdb->insert(
                            $this->table_name,
                            array(
                                'content' => wp_kses_post(substr($page, 500 * $i, 1000)),
                                'vector' => json_encode($response),
                                'post_id' => $post_id,
                                'type' => 'text',
                                'type_value' => $text_id,
                            )
                        );
                    }
                } catch (Exception $e) {
                }
            }

        }

        update_post_meta( $post_id, '_magicai_trained_text', $trained_text );

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'Text Train Data Added',
            sprintf( 'Template Name: %s, Template ID: %s, Title: %s', get_the_title( $post_id ), $post_id, $title )
        );

        wp_send_json( [
            'output' => $this->text_result_table( 
                get_post_meta( $post_id, '_magicai_trained_text', true )
            ),
            'message' => __( 'Tained Successful!', 'magicai-wp' )
        ] );

    }

    /**
     * Generate HTML table for displaying trained text data.
     * 
     * This function generates an HTML table to display the trained text data.
     *
     * @param array $data An array containing trained text data.
     * 
     * @return string HTML representation of the trained text data table.
     */
    function text_result_table( $data = array() ) {
        ob_start();
        ?>
        <div class="result">
            <h3><?php esc_html_e( 'Trained Text Files', 'magicai-wp' ); ?></h3>
            <table class="widefat striped" style="border-radius: var(--magicai-border-r);">
                <thead>
                    <th><?php esc_html_e( 'Title', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Content', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Status', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Word', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Action', 'magicai-wp' ); ?></th>
                </thead>
                <tbody>
                <?php
                    $total_word_count = 0;
                    foreach ( $data as $id => $text ) {
                        $word_count = $text['word_count'];
                        $total_word_count += $word_count;
                        printf(
                            '<tr>
                                <td>%1$s</td>
                                <td>%2$s</td>
                                <td>%3$s %4$s</td>
                                <td>%5$s</td>
                                <td>
                                    <div class="actions">
                                        <div title="%9$s" class="magicai-chatbot--train-text--edit" data-id="%7$s" data-trained="%8$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M15.7279 9.57627L14.3137 8.16206L5 17.4758V18.89H6.41421L15.7279 9.57627ZM17.1421 8.16206L18.5563 6.74785L17.1421 5.33363L15.7279 6.74785L17.1421 8.16206ZM7.24264 20.89H3V16.6473L16.435 3.21231C16.8256 2.82179 17.4587 2.82179 17.8492 3.21231L20.6777 6.04074C21.0682 6.43126 21.0682 7.06443 20.6777 7.45495L7.24264 20.89Z"></path></svg></div>
                                        <div title="%6$s" class="magicai-chatbot--train-text--delete" data-id="%7$s" data-trained="%8$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM9 11H11V17H9V11ZM13 11H15V17H13V11ZM9 4V6H15V4H9Z"></path></svg></div>
                                    </div>
                                </td>
                            </tr>',
                            wp_trim_words( $text['title'] , 10 ),
                            wp_trim_words( $text['content'] , 10 ),
                            magicai_helper()->label_help_tip( 'This data was trained. Ready to use in the chatbot.', false ),
                            '<span class="trained">' . esc_html__('Trained') . '<span class="dashicons dashicons-yes"></span></span>',
                            $text['word_count'],
                            esc_attr__( 'Delete', 'magicai-wp' ),
                            esc_attr( $id ),
                            'yes',
                            esc_attr__( 'Edit', 'magicai-wp' ),
                        );
                    }
                ?>
                </tbody>
                <tfoot>
                    <td colspan="3"><?php printf( '%s: %s', __( 'Total Text', 'magicai-wp' ), count($data) ); ?></td>
                    <td colspan="2"><?php printf( '%s: %s', __( 'Total Word', 'magicai-wp' ), $total_word_count ); ?></td>
                </tfoot>
            </table>
        </div>
        <?php 
        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;
    }

    /**
     * Delete text data.
     * 
     * This function deletes the specified text data.
     *
     * @return void
     */
    function delete_text() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $text_id = !empty( $_POST['text_id'] ) ? sanitize_text_field( $_POST['text_id'] ) : '';
        $is_trained = !empty( $_POST['is_trained'] ) ? sanitize_text_field( $_POST['is_trained'] ) : 'no';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$text_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Text File does not exists!', 'magicai-wp')
            ] );
        }

        $data = get_post_meta( $post_id, '_magicai_trained_text', true );
        if ( isset( $data[$text_id] ) ){
            unset($data[$text_id]);
            update_post_meta( $post_id, '_magicai_trained_text', $data );
        }

        if ( $is_trained == 'yes' ) {
            $this->delete_table_row( $type = 'text', $type_value = $text_id );
        }

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'Text Train Data Deleted',
            sprintf( 'Template Name: %s, Template ID: %s', get_the_title( $post_id ), $post_id )
        );

        wp_send_json( ['message' => __('Deleted', 'magicai-wp') ] );
    }

    /**
     * Get trained text data.
     * 
     * This function retrieves trained text data for a specified post and text ID.
     *
     * @return void
     */
    function get_train_text() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $text_id = !empty( $_POST['text_id'] ) ? sanitize_text_field( $_POST['text_id'] ) : '';

        if ( $post_id && $text_id ) {
            $data = get_post_meta( $post_id, '_magicai_trained_text', true );
    
            wp_send_json( [
                $post_id,
                'title' => $data[$text_id]['title'],
                'content' => $data[$text_id]['content'],
            ] );
        }

    }

    /**
     * Train with QA data.
     * 
     * This function handles the training process with QA data provided via POST request.
     * It sanitizes and validates the input data and then parses the QA data and embeds it using OpenAI.
     *
     * @return void
     */
    function train_with_qa() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $qa_id = !empty( $_POST['qa_id'] ) ? sanitize_text_field( $_POST['qa_id'] ) : '';
        $q = !empty( $_POST['q'] ) ? sanitize_text_field( $_POST['q'] ) : '';
        $a = !empty( $_POST['a'] ) ? sanitize_textarea_field( $_POST['a'] ) : '';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$q ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Question does not exists!', 'magicai-wp')
            ] );
        }

        if ( !$a ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Answer does not exists!', 'magicai-wp')
            ] );
        }

        if ( empty( $qa_id ) ) {
            $qa_id = uniqid();
        }

        $trained_qa = get_post_meta( $post_id, '_magicai_trained_qa', true );
        $this->parse_qa_data_and_embedding( $post_id, $qa_id, $q, $a, $trained_qa );

    }

    /**
     * Parse QA data and embed it.
     * 
     * This function parses the provided text data, splits it into smaller chunks, and embeds each chunk using OpenAI.
     * It then stores the embedded data in the database.
     *
     * @param int    $post_id    The ID of the post associated with the QA data.
     * @param string $qa_id      The ID of the QA data.
     * @param string $q          The title of the QA data.
     * @param string $a          The content of the QA data.
     * @param array  $trained_qa An array containing previously trained QA data.
     * 
     * @return void
     */
    function parse_qa_data_and_embedding( $post_id, $qa_id, $q, $a, $trained_qa = array() ) {

        global $wpdb;
        
        if ( !is_array( $trained_qa ) ) {
            $trained_qa = array();
        }

        $open_ai = new Orhanerday\OpenAi\OpenAi(magicai_helper()->get_option( 'openai_key' ));

        // if data exists before, delete from db
        if ( isset($trained_qa[$qa_id]) ) {
            $this->delete_table_row( $type = 'qa', $type_value = $qa_id );
        }
    
        $text = "Question: $q. Answer: $a";
        $page = $text;

        if (!mb_check_encoding($text, 'UTF-8')) {
            $page = mb_convert_encoding($text, 'UTF-8', mb_detect_encoding($text));
        } else {
            $page = $text;
        }

        $countwords = strlen($page) / 500 + 1;
        $trained_qa[$qa_id] = [
            'q' => $q,
            'a' => $a,
            'word_count' => magicai_helper()->get_word_count($text),
        ];
        
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
                        $wpdb->insert(
                            $this->table_name,
                            array(
                                'content' => wp_kses_post(substr($page, 500 * $i, strlen($page) - 500 * $i)),
                                'vector' => json_encode($response),
                                'post_id' => $post_id,
                                'type' => 'qa',
                                'type_value' => $qa_id,
                            )
                        );
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
                        $wpdb->insert(
                            $this->table_name,
                            array(
                                'content' => wp_kses_post(substr($page, 500 * $i, 1000)),
                                'vector' => json_encode($response),
                                'post_id' => $post_id,
                                'type' => 'qa',
                                'type_value' => $qa_id,
                            )
                        );
                    }
                } catch (Exception $e) {
                }
            }

        }

        update_post_meta( $post_id, '_magicai_trained_qa', $trained_qa );

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'Q&A Train Data Added',
            sprintf( 'Template Name: %s, Template ID: %s', get_the_title( $post_id ), $post_id )
        );

        wp_send_json( [
            'output' => $this->qa_result_table( 
                get_post_meta( $post_id, '_magicai_trained_qa', true )
            ),
            'message' => __( 'Tained Successful!', 'magicai-wp' )
        ] );

    }

    /**
     * Generate HTML table for displaying trained QA data.
     * 
     * This function generates an HTML table to display the trained QA data.
     *
     * @param array $data An array containing trained QA data.
     * 
     * @return string HTML representation of the trained QA data table.
     */
    function qa_result_table( $data = array() ) {
        ob_start();
        ?>
        <div class="result">
            <h3><?php esc_html_e( 'Trained Q&A Files', 'magicai-wp' ); ?></h3>
            <table class="widefat striped" style="border-radius: var(--magicai-border-r);">
                <thead>
                    <th><?php esc_html_e( 'Question', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Answer', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Status', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Word', 'magicai-wp' ); ?></th>
                    <th><?php esc_html_e( 'Action', 'magicai-wp' ); ?></th>
                </thead>
                <tbody>
                <?php
                    $total_word_count = 0;
                    foreach ( $data as $id => $qa ) {
                        $word_count = $qa['word_count'];
                        $total_word_count += $word_count;
                        printf(
                            '<tr>
                                <td>%1$s</td>
                                <td>%2$s</td>
                                <td>%3$s %4$s</td>
                                <td>%5$s</td>
                                <td>
                                    <div class="actions">
                                        <div title="%9$s" class="magicai-chatbot--train-qa--edit" data-id="%7$s" data-trained="%8$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M15.7279 9.57627L14.3137 8.16206L5 17.4758V18.89H6.41421L15.7279 9.57627ZM17.1421 8.16206L18.5563 6.74785L17.1421 5.33363L15.7279 6.74785L17.1421 8.16206ZM7.24264 20.89H3V16.6473L16.435 3.21231C16.8256 2.82179 17.4587 2.82179 17.8492 3.21231L20.6777 6.04074C21.0682 6.43126 21.0682 7.06443 20.6777 7.45495L7.24264 20.89Z"></path></svg></div>
                                        <div title="%6$s" class="magicai-chatbot--train-qa--delete" data-id="%7$s" data-trained="%8$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM9 11H11V17H9V11ZM13 11H15V17H13V11ZM9 4V6H15V4H9Z"></path></svg></div>
                                    </div>
                                </td>
                            </tr>',
                            wp_trim_words( $qa['q'] , 10 ),
                            wp_trim_words( $qa['a'] , 10 ),
                            magicai_helper()->label_help_tip( 'This data was trained. Ready to use in the chatbot.', false ),
                            '<span class="trained">' . esc_html__('Trained') . '<span class="dashicons dashicons-yes"></span></span>',
                            $qa['word_count'],
                            esc_attr__( 'Delete', 'magicai-wp' ),
                            esc_attr( $id ),
                            'yes',
                            esc_attr__( 'Edit', 'magicai-wp' ),
                        );
                    }
                ?>
                </tbody>
                <tfoot>
                    <td colspan="3"><?php printf( '%s: %s', __( 'Total Q&A', 'magicai-wp' ), count($data) ); ?></td>
                    <td colspan="2"><?php printf( '%s: %s', __( 'Total Word', 'magicai-wp' ), $total_word_count ); ?></td>
                </tfoot>
            </table>
        </div>
        <?php 
        $output_string = ob_get_contents();
        ob_end_clean();
        return $output_string;
    }

    /**
     * Delete QA data.
     * 
     * This function deletes the specified QA data.
     *
     * @return void
     */
    function delete_qa() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $qa_id = !empty( $_POST['qa_id'] ) ? sanitize_text_field( $_POST['qa_id'] ) : '';
        $is_trained = !empty( $_POST['is_trained'] ) ? sanitize_text_field( $_POST['is_trained'] ) : 'no';

        if ( !$post_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('ChatBot data is invalid!', 'magicai-wp')
            ] );
        }

        if ( !$qa_id ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Text File does not exists!', 'magicai-wp')
            ] );
        }

        $data = get_post_meta( $post_id, '_magicai_trained_qa', true );
        if ( isset( $data[$qa_id] ) ){
            unset($data[$qa_id]);
            update_post_meta( $post_id, '_magicai_trained_qa', $data );
        }

        if ( $is_trained == 'yes' ) {
            $this->delete_table_row( $type = 'qa', $type_value = $qa_id );
        }

        MagicAI_Logs::instance()->add_log(
            'chatbot-train',
            'Q&A Train Data Deleted',
            sprintf( 'Template Name: %s, Template ID: %s', get_the_title( $post_id ), $post_id )
        );

        wp_send_json( ['message' => __('Deleted', 'magicai-wp') ] );
    }

     /**
     * Get trained QA data.
     * 
     * This function retrieves trained QA data for a specified post and QA ID.
     *
     * @return void
     */
    function get_train_qa() {

        $post_id = !empty( $_POST['post_id'] ) ? intval( $_POST['post_id'] ) : '';
        $qa_id = !empty( $_POST['qa_id'] ) ? sanitize_text_field( $_POST['qa_id'] ) : '';

        if ( $post_id && $qa_id ) {
            $data = get_post_meta( $post_id, '_magicai_trained_qa', true );
    
            wp_send_json( [
                $post_id,
                'q' => $data[$qa_id]['q'],
                'a' => $data[$qa_id]['a'],
            ] );
        }

    }

    /**
     * Retrieves the most similar text data based on the provided prompt.
     * 
     * Retrieves the post ID and prompt text from the POST request, then fetches
     * the vectors associated with the post ID from the database. Calculates cosine
     * similarity between the input prompt vector and the stored vectors, sorts them
     * by similarity, and sends back the top 5 most similar texts along with an
     * extra prompt for further interaction.
     *
     * @return void
     */
    function getMostSimilarText() {

        $chat_id = !empty( $_POST['chat_id'] ) ? intval( $_POST['chat_id'] ) : '';
        $text = !empty( $_POST['prompt'] ) ? sanitize_text_field( $_POST['prompt'] ) : '';
        
        $template_id = get_post_meta( $chat_id, '_magicai_chatbot_template_id', true );
        $post_id = intval( $template_id );

		$table_name = $this->table_name;
		$sql = "SELECT * FROM $table_name WHERE post_id = {$post_id}";
        
        $vectors = $this->wpdb->get_results($sql);

        if ( !$vectors ) {
            wp_send_json( [
                'extra_prompt' => null,
            ] );
        }

        $open_ai = new Orhanerday\OpenAi\OpenAi(magicai_helper()->get_option( 'openai_key' ));
        $vector = $open_ai->embeddings([
            'model' => 'text-embedding-ada-002',
            'input' => $text
        ]);
        $vector = json_decode($vector)->data[0]->embedding;
        $similarVectors = [];

        foreach ($vectors as $v) {
            $cosineSimilarity = MagicAI_Actions::instance()->calculateCosineSimilarity($vector, json_decode($v->vector));
            $similarVectors[] = [
                'id' => $v->id,
                'content' => $v->content,
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
            'extra_prompt' => "Must not reference previous chats if user asking about content. Must reference this content if only user is asking about content. Else just response as an assistant shortly and professionaly without must not referencing this content. \n\n\n\n\nUser qusetion: $prompt \n\n\n\n\n Content: \n $result"
        ] );

    }

    /**
     * Ban an IP address.
     *
     * This function bans the provided IP address by adding it to the IP storage.
     * If the IP address is already present in the storage, it removes it.
     * The function is typically used in an AJAX request to ban or unban IP addresses.
     *
     * @since 1.4
     */
    function ban_ip() {

        // Get the IP address from the POST data.
        $ip = !empty( $_POST['ip'] ) ? sanitize_text_field( $_POST['ip'] ) : '';

        // Retrieve the IP storage.
        $ip_storage = get_option( 'magicai_ip_storage', array() );

        // If IP address is not provided, send error response.
        if ( empty( $ip ) ) {
            wp_send_json( [
                'error' => true,
                'message' => __( 'Failed to get IP', 'magicai-wp' ),
            ] );
        }

        // Check if the IP address is already banned.
        if ( magicai_helper()->check_user_ip_status( $ip, $delete = true ) ) {
            MagicAI_Logs::instance()->add_log(
                'chatbot',
                'User IP Unbanned',
                "IP: $ip"
            );
            wp_send_json( [
                'error' => false,
                'message' => __( 'IP Unban Successful', 'magicai-wp' ),
                'text' => __( 'Ban This IP', 'magicai-wp' ),
            ] );
        } else {
            // If not banned, add it to the storage.
            $ip_storage[] = $ip;
            update_option( 'magicai_ip_storage', $ip_storage );

            MagicAI_Logs::instance()->add_log(
                'chatbot',
                'User IP Banned',
                "IP: $ip"
            );

            // Send success response.
            wp_send_json( [
                'error' => false,
                'message' => __( 'IP Ban Successful', 'magicai-wp' ),
                'text' => __( 'Unban This IP', 'magicai-wp' ),
            ] );
        }
    }
    
}
new MagicAI_ChatBot();
