<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

class MagicAI_Logs {

    private $wpdb;

	private $table_name;
	const DB_VERSION = '1.0.0';

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Logs The single instance of the class.
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
	 * @return MagicAI_Logs An instance of the class.
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

		$this->db_init();
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

	}

	/**
	 * Initialize DB
	 *
	 * Prepare the `wp_magicai_logs` database table.
	 *
	 * @since 1.0.0
	 *
	 */
	public function db_init() {
		global $wpdb;
		$this->wpdb = $wpdb;

		$this->table_name = $this->wpdb->prefix . 'magicai_logs';

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
	 * Creates the `wp_magicai_logs` database table.
	 *
	 * @since 1.0.0
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
			status text null,
			message text null,
			details text null,
			total_tokens int null,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
		) {$charset_collate};";

		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		$this->wpdb->query( $sql );

		// Check if table was created successfully.
		// phpcs:disable WordPress.DB.PreparedSQL.NotPrepared
		if ( $this->wpdb->get_var( $query ) === $table_name ) {
			update_option( 'magicai_logs_db_version', self::DB_VERSION, false );
		}
	}

	/**
	 * Add Indexes
	 *
	 * Adds an index to the logs table for the creation date column.
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
	 * Empties the contents of the logs DB table.
	 *
	 * @since 1.0.0
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
	 * Delete the logs DB table.
	 *
	 * @since 1.0.0
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
	 * @since 1.0.0
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
	 * Admin Page View
	 *
	 * View logs data on the page
	 *
	 * @since 1.0.0
	 */
	public function view() {

		if ( ! current_user_can( 'manage_options' ) ) {
			return;
		}
		?>

		<div class="wrap">
			<h1><?php echo esc_html( get_admin_page_title() ); ?></h1>

			<form method="post">
			<?php
				$table = new MagicAI_Logs_Table();
				$table->prepare_items();
				$table->search_box('search', 'search_id');
				$table->display();
			?>
			</form>
			
		</div>

		<?php
	}


	/**
	 * Get and display the last N logs from the database table.
	 *
	 * This function retrieves the last N logs from the specified database table,
	 * orders them by the 'created_at' column in descending order, and displays the results
	 * in an HTML table format. The table includes columns for 'Avatar', 'Status', 'Message', and 'Date'.
	 *
	 * @param int $count The number of last logs to retrieve and display. Default is 5.
	 *
	 * @return void
	 */
	public function get_last_logs( $count = 5, $atts = null ) {

		$table_name = $this->table_name;
		$sql = "SELECT * FROM $table_name ORDER BY created_at DESC LIMIT $count";

		if ( isset( $atts['user_filter'] ) ) {
			$filter = $atts['user_filter'];
			if ( ( $filter == 'true' || !empty( $filter ) ) && $filter != 'false' ) {
				$user_id = $filter == 'true' ? get_current_user_id() : intval($filter);
				$sql = "SELECT * FROM $table_name WHERE user_id = $user_id ORDER BY created_at DESC LIMIT $count";
			}
		}
		
		$results = $this->wpdb->get_results($sql);

		if ( empty( $results ) ) {
			return printf( '<p>%s</p>', esc_html__('There is no log record yet.', 'magicai-wp') );
		}

		$html = sprintf( 
			'<table>
			<tr>
				<th>%s</th>
				<th>%s</th>
				<th>%s</th>
				<th>%s</th>
			</tr>', 
			__( 'User', 'magicai-wp' ), 
			__( 'Status', 'magicai-wp' ), 
			__( 'Message', 'magicai-wp' ), 
			__( 'Date', 'magicai-wp' ), 
		);
		foreach ($results as $result) {
			$html .= sprintf(
				'<tr class="%5$s">
					<td class="user">%1$s<div class="display-name">%6$s</div></td>
					<td class="status"><span>%2$s</span></td>
					<td class="message"><span>%3$s</span></td>
					<td class="date"><span>%4$s</span></td>
				</tr>',
				get_avatar( $result->user_id, 32 ),
				$result->status,
				$result->message,
				$result->created_at,
				magicai_helper()->get_log_status_classname_by_message( $result->message ),
				get_userdata( $result->user_id )->display_name,
			);
		}

		$html .= '</table>';

		echo $html;
	}	


}
MagicAI_Logs::instance();

// WP_List_Table sınıfını dahil edin
if (!class_exists('WP_List_Table')) {
    require_once ABSPATH . 'wp-admin/includes/class-wp-list-table.php';
}

// @see https://supporthost.com/wp-list-table-tutorial/
// @see https://developer.wordpress.org/reference/classes/wp_list_table/
class MagicAI_Logs_Table extends WP_List_Table {

	private $table_data;

    function get_columns()
    {
        $columns = array(
                'cb' => '<input type="checkbox" />',
                'user_id' => esc_html__( 'User', 'magicai-wp' ),
                'ip' => esc_html__( 'IP', 'magicai-wp' ),
                'status' => esc_html__( 'Status', 'magicai-wp' ),
                'message' => esc_html__( 'Message', 'magicai-wp' ),
                'details' => esc_html__( 'Details', 'magicai-wp' ),
                'total_tokens' => esc_html__( 'Total Tokens', 'magicai-wp' ),
                'created_at' => esc_html__( 'Date', 'magicai-wp' ),
        );
        return $columns;
    }

	// Bind table with columns, data and all
	function prepare_items()
	{

		//data
        if ( isset($_POST['s']) ) {
            $this->table_data = $this->get_table_data($_POST['s']);
        } else {
            $this->table_data = $this->get_table_data();
        }

		$columns = $this->get_columns();
		$hidden = array();
		$sortable = $this->get_sortable_columns();
		$primary  = 'name';
		$this->_column_headers = array($columns, $hidden, $sortable, $primary);

		usort($this->table_data, array(&$this, 'usort_reorder'));
		
		$per_page = 10;
		$current_page = $this->get_pagenum();
		$total_items = count($this->table_data);

		$this->table_data = array_slice($this->table_data, (($current_page - 1) * $per_page), $per_page);

		$this->set_pagination_args(array(
				'total_items' => $total_items, // total number of items
				'per_page'    => $per_page, // items to show on a page
				'total_pages' => ceil( $total_items / $per_page ) // use ceil to round up
		));

		$this->items = $this->table_data;
	}

	 // Get table data
	 private function get_table_data( $search = '' ) {
        global $wpdb;

        $table = $wpdb->prefix . 'magicai_logs';

        if ( !empty($search) ) {
            return $wpdb->get_results(
                "SELECT * from {$table} WHERE 
				user_id Like '%{$search}%' OR
				ip Like '%{$search}%' OR
				status Like '%{$search}%' OR
				message Like '%{$search}%' OR
				details Like '%{$search}%'",
                ARRAY_A
            );
        } else {
            return $wpdb->get_results(
                "SELECT * from {$table}",
                ARRAY_A
            );
        }
    }

	function column_default($item, $column_name)
    {
          switch ($column_name) {
                case 'user_id':
					$user_info = get_userdata($item[$column_name]);
					if ($user_info) {
						$user_image = get_avatar($user_info->ID, 32);
						$user_name = $user_info->display_name;
						return "<div class='magicai-log-userid'>{$user_image} {$user_name}</div>";
					} else {
						return 'User not found';
					}
					break;
                case 'ip':
                case 'status':
                case 'message':
                case 'details':
                case 'total_tokens':
                case 'created_at':
                default:
                    return $item[$column_name];
          }
    }

	function column_cb($item)
    {
        return sprintf(
                '<input type="checkbox" name="element[]" value="%s" />',
                $item['id']
        );
    }

	protected function get_sortable_columns()
	{
		$sortable_columns = array(
				'ip'  => array('ip', false),
				'status' => array('status', false),
				'user_id'   => array('user_id', true),
				'total_tokens'   => array('total_tokens', true),
				'created_at'   => array('created_at', true)
		);
		return $sortable_columns;
	}

	// Sorting function
    function usort_reorder($a, $b)
    {
        // If no sort, default to user_login
        $orderby = (!empty($_GET['orderby'])) ? $_GET['orderby'] : 'created_at';

        // If no order, default to desc
        $order = (!empty($_GET['order'])) ? $_GET['order'] : 'desc';

        // Determine sort order
        $result = strcmp($a[$orderby], $b[$orderby]);

        // Send final sort direction to usort
        return ($order === 'asc') ? $result : -$result;
    }

}
