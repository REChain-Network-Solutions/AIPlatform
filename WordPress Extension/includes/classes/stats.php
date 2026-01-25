<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

/**
 * Class MagicAI_Stats
 *
 * @since 1.4
 */
class MagicAI_Stats {

    private $wpdb;

	private $table_name;
	const DB_VERSION = '1.4';

    /**
	 * Instance
	 *
	 * @since 1.4
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Stats The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @since 1.4
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_Stats An instance of the class.
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
	 * @since 1.4
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
	 * @since 1.4
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
     * @since 1.4
     *
     * @access public
     */
	public function hooks() {

	}

	/**
	 * Initialize DB
	 *
	 * Prepare the `wp_magicai_stats` database table.
	 *
	 * @since 1.4
	 *
	 */
	public function db_init() {
		global $wpdb;
		$this->wpdb = $wpdb;

		$this->table_name = $this->wpdb->prefix . 'magicai_stats';

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
	 * Creates the `wp_magicai_stats` database table.
	 *
	 * @since 1.4
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
			ai text null,
			count int null,
			is_frontend boolean DEFAULT false NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
		) {$charset_collate};";

		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		$this->wpdb->query( $sql );

		// Check if table was created successfully.
		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		if ( $this->wpdb->get_var( $query ) === $table_name ) {
			update_option( 'magicai_stats_db_version', self::DB_VERSION, false );
		}
	}

	/**
	 * Add Indexes
	 *
	 * Adds an index to the table for the creation date column.
	 *
	 * @since 1.4
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
	 * @since 1.4
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
	 * @since 1.4
	 */
	public static function drop_table() {
		global $wpdb;

		$table_name = $this->table_name;
		// Drop table
		// phpcs:disable WordPress.DB.PreparedSQL.InterpolatedNotPrepared
		$wpdb->query("DROP TABLE IF EXISTS {$table_name}");
	}

	/**
	 * Adds statistics data to the database.
	 *
	 * This method inserts statistics data into the database table.
	 * 
	 * @since 1.4
	 *
	 * @param array $data {
	 *     Optional. An array of data to be inserted into the database.
	 *
	 *     @type string $type The type of statistic.
	 *     @type string $ai The AI related to the statistic.
	 *     @type int $count The count related to the statistic.
	 * }
	 */
	public function add_stats( $data = array() ) {
        global $wpdb;

		$user_id = get_current_user_id();
		$ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];
		$is_frontend = magicai_helper()->request_is_frontend_ajax();

		$type = $data['type'] ?? '';
		$ai = $data['ai'] ?? '';
		$count = $data['count'] ?? 0;

        $wpdb->insert(
            $this->table_name,
            array(
                'user_id' => $user_id,
                'ip' => $ip,
                'type' => $type,
                'ai' => $ai,
                'count' => $count,
                'is_frontend' => $is_frontend,
            )
        );
    }


	/**
	 * Retrieves statistics results from the database.
	 *
	 * This method retrieves statistics results from the database based on the specified type and time period.
	 * 
	 * @since 1.4
	 *
	 * @param string $type Optional. The type of statistics to retrieve. Default is ''.
	 * @param int $day Optional. The number of days to consider for the statistics. Default is 7.
	 * @return array An array containing the statistics results.
	 */
	function get_results( $type = '', $day = 7 ) {
		global $wpdb;
	
		// Calculate start and end dates for the last $day days
		$start_date = date('Y-m-d', strtotime("-$day days"));
		$end_date = date('Y-m-d');
	
		// Prepare SQL query to fetch results within the date range
		$query = $wpdb->prepare("
			SELECT * 
			FROM {$this->table_name} 
			WHERE DATE(created_at) BETWEEN %s AND %s
		", $start_date, $end_date);
	
		// Execute the query and fetch results
		$results = $wpdb->get_results($query);
	
		// If results are found, group them by date
		if ($results) {
			$grouped_results = $final_results = array(); // Initialize an empty array to group results by date

			if ( $type == 'generator' ) {

				foreach ($results as $result) {

					$ai = $result->ai;
					if (!isset($final_results[$ai])) {
						$final_results[$ai] = [
							'count' => 0
						];
					}
					$final_results[$ai]['count'] += 1;
				}


			} elseif ( $type == 'is_frontend' ) {

				$frontend = $backend = 0;
				foreach ($results as $result) {
					$result->is_frontend ? $frontend +=1 : $backend +=1;
				}
				
				$final_results = [
					'frontend' => $frontend,
					'backend' => $backend
				];

			} else {
				foreach ($results as $result) {
					$date = date('d.m', strtotime($result->created_at)); // Get the date
					$grouped_results[$date][] = $result; // Store the result in the array grouped by date
				}
	
				// Process the grouped results
				foreach ($grouped_results as $date => $group) {
					foreach ($group as $item) {
						$type = $item->type;
						if (!isset($final_results[$date][$type])) {
							$final_results[$date][$type] = [
								'count' => 0
							];
						}
						$final_results[$date][$type]['count'] += $item->count;
					}
				}
			}
	

			return $final_results;
		
		}
	}
	

	/**
	 * Check user attempts for a specific type within a certain time frame.
	 *
	 * This function checks the number of attempts made by a user for a specific type
	 * within a certain time frame. It updates the attempt count and last attempt time
	 * in the database accordingly.
	 *
	 * @since 1.4
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
MagicAI_Stats::instance();