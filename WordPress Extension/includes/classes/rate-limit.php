<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * Class MagicAI_RateLimit
 *
 * @since 1.3
 */
class MagicAI_RateLimit {

    private $wpdb;

	private $table_name;
	const DB_VERSION = '1.0.0';

    /**
	 * Instance
	 *
	 * @since 1.3
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_RateLimit The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @since 1.3
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_RateLimit An instance of the class.
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
	 * @since 1.3
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
	 * @since 1.3
	 *
	 * @access public
	 */
    public function init() {

		$this->db_init();
		$this->hooks();

    }

    /**
     * Hooks Registration
     *
     * This function registers WordPress action and filter hooks for the MagicAI plugin.
     * It allows you to define custom actions and filters to extend or modify the plugin's functionality.
     *
     * @since 1.3
     *
     * @access public
     */
	public function hooks() {

	}

	/**
	 * Initialize DB
	 *
	 * Prepare the `wp_magicai_rate_limit` database table.
	 *
	 * @since 1.3
	 *
	 */
	public function db_init() {
		global $wpdb;
		$this->wpdb = $wpdb;

		$this->table_name = $this->wpdb->prefix . 'magicai_rate_limit';

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
	 * Creates the `wp_magicai_rate_limit` database table.
	 *
	 * @since 1.3
	 *
	 * @param string $query to that looks for the Events table in the DB. Used for checking if table was created.
	 */
	private function create_table( $query ) {
		require_once ABSPATH . 'wp-admin/includes/upgrade.php';

		$table_name = $this->table_name;
		$charset_collate = $this->wpdb->get_charset_collate();

		$sql = "CREATE TABLE `{$table_name}` (
			id bigint(20) unsigned auto_increment primary key,
			user_id text null,
			ip text null,
			type text null,
			attempts int null,
			last_attempt TIMESTAMP null,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
		) {$charset_collate};";

		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		$this->wpdb->query( $sql );

		// Check if table was created successfully.
		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		if ( $this->wpdb->get_var( $query ) === $table_name ) {
			update_option( 'magicai_rate_limit_db_version', self::DB_VERSION, false );
		}
	}

	/**
	 * Add Indexes
	 *
	 * Adds an index to the table for the creation date column.
	 *
	 * @since 1.0.0
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
	 *
	 * @since 1.3
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
	 *
	 * @since 1.3
	 */
	public static function drop_table() {
		global $wpdb;

		$table_name = $this->table_name;
		// Drop table
		// phpcs:disable WordPress.DB.PreparedSQL.InterpolatedNotPrepared
		$wpdb->query("DROP TABLE IF EXISTS {$table_name}");
	}


	/**
	 * Add Log
	 *
	 * Adds a log entry to the database.
	 *
	 * This function is used to record events or activities in the system.
	 *
	 * @since 1.3
	 *
	 * @param string      $status       The status of the log entry (e.g., 'success', 'error', 'info').
	 * @param string      $message      A descriptive message for the log entry.
	 * @param string|null $details      Additional details or context for the log entry (optional).
	 * @param int         $total_tokens The total number of tokens associated with the log entry (default: 0).
	 *
	 * @global wpdb $wpdb WordPress database access object.
	 *
	 * @return void
	 */
	public function add_log( $status, $message, $details = null, $total_tokens = 0 ) {
        global $wpdb;

		$user_id = get_current_user_id();
		$ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];

        $wpdb->insert(
            $this->table_name,
            array(
                'user_id' => $user_id,
                'ip' => $ip,
                'status' => $status,
                'message' => $message,
                'details' => $details,
                'total_tokens' => $total_tokens,
            )
        );
    }

	/**
	 * Check user attempts for a specific type within a certain time frame.
	 *
	 * This function checks the number of attempts made by a user for a specific type
	 * within a certain time frame. It updates the attempt count and last attempt time
	 * in the database accordingly.
	 *
	 * @since 1.3
	 *
	 * @param string $type The type for which attempts are being checked.
	 * @return bool True if the attempt is within the limit and the database is updated successfully, otherwise false.
	 */
	function check( $type ) {
		
		global $wpdb;
		$table_name = $this->table_name;
		$user_id = get_current_user_id();
		$ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];
		$chatbot_options = get_option('magicai_chatbot_settings', array() );
		$limit = (isset( $chatbot_options['limit'] ) && $chatbot_options['limit']) ? intval($chatbot_options['limit']) : 0;
		$limit_per_seconds = (isset( $chatbot_options['limit_per_seconds'] ) && $chatbot_options['limit_per_seconds']) ? intval($chatbot_options['limit_per_seconds']) : 300;

		// Don't save to db if no limit.
		if ( $limit == 0 ) {
			return true;
		}

		// Prepare the query
		$query = $wpdb->prepare( "SELECT * FROM $table_name WHERE type = %s AND ip = %s", $type, $ip );
		$row = $wpdb->get_row( $query );

		// If data exists
		if ( $row ) {
			$current_time = current_time( 'timestamp' );
			$last_attempt_time = strtotime( $row->last_attempt );
			$created_at_time = strtotime( $row->created_at );
			$time_diff = $current_time - $last_attempt_time;
			
			if ( $row->attempts < $limit && $time_diff < $limit_per_seconds ) {
				// Increase the attempt
				$updated_attempts = $row->attempts + 1;
				$data = array(
					'attempts' => $updated_attempts,
					'last_attempt' => current_time( 'mysql' ),
				);
				$where = array( 'ip' => $ip );
				$update = $wpdb->update( $table_name, $data, $where );
				return true;
			} elseif( $row->attempts <= $limit && $time_diff > $limit_per_seconds ) {
				// Reset attempt
				$data = array(
					'attempts' => 1,
					'last_attempt' => current_time( 'mysql' ),
				);
				$where = array( 'ip' => $ip );
				$update = $wpdb->update( $table_name, $data, $where );
				return true;
			} else {
				// Wait for the next cycle ($limit_per_seconds)
				return false;
			}
		} else {
			// Create new row
			$wpdb->insert(
				$this->table_name,
				array(
					'user_id' => $user_id,
					'ip' => $ip,
					'type' => $type,
					'attempts' => 1,
					'last_attempt' => current_time( 'mysql' ),
				)
			);
			return true;
		}

	}

}
MagicAI_RateLimit::instance();