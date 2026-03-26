<?php

defined('BASEPATH') or exit('No direct script access allowed');
/**
 * Class Report_builder
 */
class Report_builder extends AdminController
{
	/**
	 * __construct
	 */
	public function __construct()
	{
		parent::__construct();
		$this->load->model('report_builder_model');
		hooks()->do_action('report_builder_init');

	}

	/**
	 * setting
	 * @return [type] 
	 */
	public function setting()
	{
		if (!has_permission('report_builder', '', 'edit') && !is_admin() && !has_permission('report_builder', '', 'create')) {
			access_denied('report_builder');
		}
		$this->load->model('staff_model');
		$this->load->model('departments_model');
		$this->load->model('roles_model');

		$data['group'] = $this->input->get('group');
		$data['title'] = _l('setting');

		$data['tab'][] = 'general_setting';
		$data['tab'][] = 'category';


		if ($data['group'] == '') {
			$data['group'] = 'general_setting';
		}elseif ($data['group'] == 'general_setting') {

		}elseif($data['group'] == 'category'){
		}

		$data['report_title_sample'] = _l('report').'-'.date('M-Y');
		$data['get_report_setting'] = $this->report_builder_model->get_report_setting();

		$data['categories'] = $this->report_builder_model->get_category();
		$data['staffs'] = $this->staff_model->get();
		$data['departments'] = $this->departments_model->get();
		$data['roles']         = $this->roles_model->get();
		$data['general_setting_href']         = 'general_setting';


		$data['tabs']['view'] = 'settings/' . $data['group'];
		
		$this->load->view('settings/manage_setting', $data);
	}

	/**
	 * report manage
	 * @return [type] 
	 */
	public function report_manage($check_delete = 0)
	{
	    if (!has_permission('report_builder', '', 'view') ) {
			access_denied('report');
		}

		$data['title'] = _l('reports');
		$data['categories'] = $this->report_builder_model->get_category();
		$data['staffs'] = $this->staff_model->get();


		$this->load->view('report_builder/reports/report_management', $data);
	}

	/**
	 * report table
	 * @return [type] 
	 */
	public function report_table()
	{
		$this->app->get_table_data(module_views_path('report_builder', 'reports/report_table'));
	}

	/**
	 * add data source
	 * @param string $id 
	 */
	public function add_data_source($id='')
	{	

		if (!has_permission('report_builder', '', 'edit') && !is_admin() && !has_permission('report_builder', '', 'create')) {
			access_denied('report_builder');
		}

		$add_report_setting = $this->report_builder_model->add_report_setting();

		if($add_report_setting == 'need_add_general_setting_before_create_report'){
			set_alert('warning', _l($add_report_setting));
			redirect(admin_url('report_builder/report_manage'));
		}else{
			if($add_report_setting){
				$data['report_template_id'] = $add_report_setting;
				redirect(admin_url('report_builder/data_source_manage/'.$add_report_setting));

			}else{
				set_alert('warning', _l('add_setting_for_report_failed'));
				redirect(admin_url('report_builder/report_manage'));
			}
		}

	}

	/**
	 * add data source
	 */
	public function data_source_manage($id='')
	{	
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('report_builder');
		}

		$data=[];
		$data['title'] = _l('rb_add_data_source');

		$data['header'] = _l('rb_data_source');

		$data['group'] = $this->input->get('group');
		$data['tab'][] = 'relationships';
		$data['tab'][] = 'filters';

		if($data['group'] == ''){
			$data['group'] = 'relationships';
		}

		$relationship_row_template='';
		if($data['group'] == 'relationships'){

			$relationship_modal_get_realated_table = $this->report_builder_model->relationship_modal_get_realated_table($id);
			$data['tables'] = $relationship_modal_get_realated_table;
			$relationship_data = $this->report_builder_model->get_data_source_relationship_by_template_id($id);
			$list_table_temp = [];
			if(count($relationship_data) > 0){
				$list_table = [];
				//update: > 2 row
				foreach ($relationship_data as $key => $relationship) {
					$list_table[] = $relationship['left_table'];

					$table_type_of_join_data = [];
					$table_type_of_join_data = [
						'array_table_names' => $list_table,
						'table_names' => $relationship['left_table'],
					];


					$get_table_type_of_join_data = $this->report_builder_model->relationship_get_table_type_of_join_data($table_type_of_join_data);

					if($key+1 == count($relationship_data)){
						$list_table_temp = $get_table_type_of_join_data['related_table_data'];
					}

					$main_table = false;
					if($key == 0){
						$main_table = true;
					}

					$relationship_row_template .= $this->report_builder_model->create_relationship_row_template($get_table_type_of_join_data['related_table_data'], $get_table_type_of_join_data['select_data_type_of_join'], $relationship['left_table'], $relationship['query_string'], $relationship['join_type'], $relationship['id'], 'items[' . $relationship['id'] . ']', $main_table);
				}

			}else{
				//insert: 2 row, 1 main, 1 join table
				$relationship_row_template .= $this->report_builder_model->create_relationship_row_template($data['tables'], [], '', '', '', '', 'newitems', true);

			}

			$list_table = count($list_table_temp) > 0 ? $list_table_temp : $data['tables'];
			$relationship_row_template .= $this->report_builder_model->create_relationship_row_template($list_table, [], '', '', '', '', '', false, false);

		}

		$data['tabs']['view'] = 'templates/data_sources/'.$data['group'];
		$data['report_template_id'] = $id;
		$data['relationship_row_template'] = $relationship_row_template;

		$this->load->view('report_builder/templates/data_sources/data_source', $data);
	}

	/**
	 * relationship table
	 * @return [type] 
	 */
	public function relationship_table()
	{
		$this->app->get_table_data(module_views_path('report_builder', 'templates/data_sources/relationship_table'));
	}

	/**
	 * relationship modal
	 * @return [type] 
	 */
	public function relationship_modal()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}
		$template_data=$this->input->post();
		$template_id=$template_data['template_id'];

		$data=[];
		$relationship_modal_get_realated_table = $this->report_builder_model->relationship_modal_get_realated_table($template_id);
		$data['tables'] = $relationship_modal_get_realated_table;
		$data['template_id'] = $template_id;

		$this->load->view('templates/data_sources/relationship_modal', $data);
	}

	/**
	 * get table related data
	 * @param  [type] $table 
	 * @return [type]        
	 */
	public function get_table_related_data($table)
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}

		$table_related_data = $this->report_builder_model->get_table_related_data($table);

		echo json_encode([
			'left_field_option_1' => $table_related_data['left_field_option_1'],
			'left_field_option_2' => $table_related_data['left_field_option_2'],
			'right_field_option_1' => $table_related_data['right_field_option_1'],
			'right_field_option_2' => $table_related_data['right_field_option_2'],
			'right_table' => $table_related_data['right_table'],
		]);
	}

	/**
	 * add relationship
	 * @param string $id 
	 */
	public function add_relationship($id='')
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('data_source');
		}

		if ($this->input->post()) {
			$data = $this->input->post();
			$result = $this->report_builder_model->add_relationship($data);

			if($result){
				$status = 'true';
				$message = _l('rb_added_successfully');
				set_alert('success', _l('rb_added_successfully'));

			}else{
				$status = 'false';
				$message = _l('rb_added_failed');
				set_alert('warning', _l('rb_added_failed'));

			}

			redirect(admin_url('report_builder/data_source_manage/'.$data['templates_id'].'?group=relationships'));
			
		}
	}

	/**
	 * relationship
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_relationship($id, $template_id)
	{
	    if (!has_permission('report_builder', '', 'delete')  && !is_admin()) {
			access_denied('data_source');
		}

		$success = $this->report_builder_model->delete_relationship($id);
		if ($success) {
			set_alert('success', _l('rb_deleted'));
		} else {
			set_alert('warning', _l('problem_deleting'));
		}
		redirect(admin_url('report_builder/data_source_manage/'.$template_id.'?group=relationships'));
	}

	/**
	 * category setting
	 * @param  string $id 
	 * @return [type]     
	 */
	public function category_setting($id = '')
	{
		if ($this->input->post()) {
			$message = '';
			$data = $this->input->post();

			if (!$this->input->post('id')) {

				$mess = $this->report_builder_model->add_category($data);
				if ($mess) {
					set_alert('success', _l('rb_added_successfully'));

				} else {
					set_alert('warning', _l('rb_added_failed'));
				}
				redirect(admin_url('report_builder/setting?group=category'));

			} else {
				$id = $data['id'];
				unset($data['id']);
				$success = $this->report_builder_model->update_category($data, $id);
				if ($success) {
					set_alert('success', _l('rb_updated_successfully'));
				} else {
					set_alert('warning', _l('rb_updated_failed'));
				}

				redirect(admin_url('report_builder/setting?group=category'));
			}
		}
	}

	/**
	 * delete category
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_category($id)
	{
		if (!$id) {
			redirect(admin_url('report_builder/setting?group=category'));
		}

		if(!has_permission('report_builder', '', 'delete')  &&  !is_admin()) {
			access_denied('report_builder');
		}

		$response = $this->report_builder_model->delete_category($id);
		if ($response) {
			set_alert('success', _l('deleted'));
			redirect(admin_url('report_builder/setting?group=category'));
		} else {
			set_alert('warning', _l('problem_deleting'));
			redirect(admin_url('report_builder/setting?group=category'));
		}
	}

	/**
	 * general setting
	 * @return [type] 
	 */
	public function general_setting()
	{
		if (!has_permission('report_builder', '', 'edit') && !is_admin() && !has_permission('report_builder', '', 'create')) {
			access_denied('report_builder');
		}

		$data = $this->input->post();

		if ($data) {
			$success = $this->report_builder_model->update_report_general_setting($data);

			if ($success == true) {
				$message = _l('rb_updated_successfully');
				set_alert('success', $message);
			}
			redirect(admin_url('report_builder/setting?group=general_setting'));
		}
	}

	/**
	 * filter table
	 * @return [type] 
	 */
	public function filter_table()
	{
		$this->app->get_table_data(module_views_path('report_builder', 'templates/data_sources/filter_table'));
	}

	/**
	 * filter modal
	 * @return [type] 
	 */
	public function filter_modal()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}
		$template_data=$this->input->post();
		$template_id=$template_data['template_id'];
		$slug=$template_data['slug'];
		$datasource_filter_id=$template_data['datasource_filter_id'];

		$data=[];

		if($slug == 'update'){
			$datasource_filter = $this->report_builder_model->get_datasource_filter($datasource_filter_id);
			$data['datasource_filter'] = $datasource_filter;
			$data['datasource_filter_id'] = $datasource_filter_id;
			$data['field_name_option'] = $this->report_builder_model->table_get_list_fields($datasource_filter->table_name, true);

			$table_name = '';
			$field_name = '';
			if(isset($datasource_filter)){
				$table_name = $datasource_filter->table_name;
				$field_name = $datasource_filter->field_name;
			}
			$data['get_filter_type'] = $this->report_builder_model->get_filter_type($table_name, $field_name, '', false);

		}

		$relationship_modal_get_realated_table = $this->report_builder_model->filter_modal_get_table($template_id);
		$data['tables'] = $relationship_modal_get_realated_table;
		$data['template_id'] = $template_id;

		$this->load->view('templates/data_sources/filter_modal', $data);
	}

	/**
	 * get list fields
	 * @param  [type] $table 
	 * @return [type]        
	 */
	public function get_list_fields($table)
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}

		$list_field_options = '';
		if($table != ''){
			$list_field_options = $this->report_builder_model->table_get_list_fields($table);
		}

		echo json_encode([
			'list_field_options' => $list_field_options,
		]);
	}

	/**
	 * add filter
	 * @param string $id 
	 */
	public function add_filter($id='')
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('data_source');
		}

		if ($this->input->post()) {

			$formdata = $this->input->post();
			$data=[];
			foreach ($formdata['formdata'] as $value) {
				if($value['name'] != 'csrf_token_name' && $value['name'] != 'filter_value_1[]'){
					$data[$value['name']] = $value['value'];
				}elseif($value['name'] == 'filter_value_1[]'){
					$data['filter_value_1'][] = $value['value'];
				}
			}

			if(isset($data['filter_value_1']) && is_array($data['filter_value_1'])){
				$data['filter_value_1'] = implode(',', $data['filter_value_1']);
			}

			if(!isset($data['filter_value_1'])){
				$data['filter_value_1'] = null;
			}

			if(!isset($data['filter_value_2'])){
				$data['filter_value_2'] = null;
			}

			$result = $this->report_builder_model->add_filter($data, $id);

			if($result){
				$status = 'true';
				$message = _l('rb_added_successfully');
			}else{
				$status = 'false';
				$message = _l('rb_added_failed');
			}

			echo json_encode([
				'status' => $status,
				'message' => $message,
			]);
		}
	}

	/**
	 * delete filter
	 * @param  [type] $id          
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function delete_filter($id, $template_id)
	{
	    if (!has_permission('report_builder', '', 'delete')  && !is_admin()) {
			access_denied('data_source');
		}

		$success = $this->report_builder_model->delete_filter($id);
		if ($success) {
			set_alert('success', _l('rb_deleted'));
		} else {
			set_alert('warning', _l('problem_deleting'));
		}
		redirect(admin_url('report_builder/data_source_manage/'.$template_id.'?group=filters'));
	}

	/**
	 * get next step report
	 * @param  [type] $template_id 
	 * @param  [type] $type        
	 * @return [type]              
	 */
	public function get_next_step_report($template_id, $type)
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}

		$results = $this->report_builder_model->get_next_step_report($template_id, $type);

		echo json_encode([
			'status' => $results['status'],
			'next_link' => $results['next_link'],
		]);
	}

	/**
	 * column manage
	 * @param  string $id 
	 * @return [type]     
	 */
	public function column_manage($id='')
	{	
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('report_builder');
		}

		$data=[];
		$data['title'] = _l('rb_add_column');
		$data['header'] = _l('rb_columns');

		$column_group_by_from_template = $this->report_builder_model->get_column_group_by_from_template_id($id);
		$data['group_by_columns'] = $column_group_by_from_template['group_by_columns'];

		$data['group'] = $this->input->get('group');


		$data['tab'][] = 'column';
		$data['tab'][] = 'label_cell_type';
		$data['tab'][] = 'aggregation_function';

		if($data['group'] == ''){
			$data['group'] = 'columns';
			$column_from_template_id = $this->report_builder_model->get_column_from_template_id($id);
			$data['column_of_table'] = $column_from_template_id['columns'];
			$data['selected_columns'] = $column_from_template_id['selected_columns'];

		}

		if($data['group'] == 'column'){
			$column_from_template_id = $this->report_builder_model->get_column_from_template_id($id);
			$data['column_of_table'] = $column_from_template_id['columns'];
			$data['selected_columns'] = $column_from_template_id['selected_columns'];

		}elseif($data['group'] == 'label_cell_type'){
			$data['columns'] = json_encode($this->report_builder_model->get_selected_column($id));

		}elseif($data['group'] == 'aggregation_function'){

			$templates_selected_column = $this->report_builder_model->get_selected_column($id);

			$data['templates_selected_column'] = $templates_selected_column;
			$data['aggregation_function'] = $this->report_builder_model->get_aggregation_function($id);

		}

		$data['rb_cell_type'] = rb_cell_type();

		$data['tabs']['view'] = 'templates/columns/'.$data['group'];

		$data['report_template_id'] = $id;
		$this->load->view('report_builder/templates/columns/column_manage', $data);
	}

	/**
	 * add columns
	 * @param string $id 
	 */
	public function add_columns($id='')
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('data_source');
		}

		if ($this->input->post()) {

			$data = $this->input->post();
			$result = $this->report_builder_model->add_columns($data, $id);

			if($result){
				set_alert('success', _l('rb_added_successfully'));
			}

			redirect(admin_url('report_builder/column_manage/'.$id.'?group=column'));
		}
	}

	/**
	 * add label cell type
	 */
	public function add_label_cell_type($id='')
	{
		if ($this->input->post()) {
			$data = $this->input->post();

			$mess = $this->report_builder_model->add_label_cell_type($data);
			if ($mess) {
				set_alert('success', _l('rb_added_successfully'));
			}
			redirect(admin_url('report_builder/column_manage/'.$id.'?group=label_cell_type'));
		}
	}

	/**
	 * { cell formatting }
	 */
	public function cell_formatting($id = ''){
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('report_builder');
		}

		$data=[];
		$data['title'] = _l('rb_cell_formatting');
		$data['header'] = _l('rb_cell_formatting');


		$data['report_template_id'] = $id;
		$data['fields'] = get_fields_by_template_id($id, true);

		$this->load->view('report_builder/templates/cells/cell_format_manage', $data);
	}

	/**
	 * cell formatting table
	 * @return [type] 
	 */
	public function cell_format_table()
	{
		$this->app->get_table_data(module_views_path('report_builder', 'templates/cells/cell_format_table'));
	}


	/**
	 * add columns
	 * @param string $id 
	 */
	public function add_cell_formatting()
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('cell_formatting');
		}


		if ($this->input->post()) {

			$formdata = $this->input->post();
			$data=[];
			foreach ($formdata['formdata'] as $value) {
				if($value['name'] != 'csrf_token_name' && $value['name'] != 'filter_value_1[]'){
					$data[$value['name']] = $value['value'];
				}elseif($value['name'] == 'filter_value_1[]'){
					$data['filter_value_1'][] = $value['value'];
				}
			}


			if(isset($data['filter_value_1']) && is_array($data['filter_value_1'])){
				$data['filter_value_1'] = implode(',', $data['filter_value_1']);
			}

			if(!isset($data['filter_value_1'])){
				$data['filter_value_1'] = null;
			}

			if(!isset($data['filter_value_2'])){
				$data['filter_value_2'] = null;
			}


			if(!$data['id']){

				$result = $this->report_builder_model->add_cell_formatting($data);

				if($result){
					$status = 'true';
					$message = _l('rb_added_successfully');
				}else{
					$status = 'false';
					$message = _l('rb_added_failed');
				}
			}else{
				$id = $data['id'];
				unset($data['id']);

				$result = $this->report_builder_model->update_cell_formatting($data, $id);

				if($result){
					$status = 'true';
					$message = _l('rb_updated_successfully');
				}
			}


			echo json_encode([
				'status' => $status,
				'message' => $message,
			]);
		}
	}

	/**
	 * relationship
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_cell_formatting($id, $template_id)
	{
	    if (!has_permission('report_builder', '', 'delete')  && !is_admin()) {
			access_denied('cell_formatting');
		}

		$success = $this->report_builder_model->delete_cell_formatting($id);
		if ($success) {
			set_alert('success', _l('rb_deleted'));
		} else {
			set_alert('warning', _l('problem_deleting'));
		}
		redirect(admin_url('report_builder/cell_formatting/'.$template_id));
	}

	/**
	 * cell formatting modal
	 * @return [type] 
	 */
	public function cell_formatting_modal()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}
		$template_data=$this->input->post();
		$template_id=$template_data['template_id'];

		$data=[];

		if($template_data['slug'] == 'edit'){
			$data['cell_formatting'] = $this->report_builder_model->get_cell_formatting($template_data['cell_formatting_id']);

			$table_name = '';
			$field_name = '';
			if(isset($data['cell_formatting'])){
				$table_name = $data['cell_formatting']->table_name;
				$field_name = $data['cell_formatting']->field_name;
			}

			$data['get_filter_type'] = $this->report_builder_model->get_filter_type($table_name, $field_name, '', false);

		}


		$data['template_id'] = $template_id;
		$data['fields'] = get_fields_by_template_id($template_id, true);


		$this->load->view('templates/cells/cell_modal', $data);
	}

	/**
	 * group manage
	 * @param  string $id 
	 * @return [type]     
	 */
	public function group_manage($id='')
	{	
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('report_builder');
		}

		$data=[];
		$data['title'] = _l('rb_groups');
		$data['header'] = _l('rb_groups');

		$data['group'] = $this->input->get('group');
		$data['tab'][] = 'group_by';
		$data['tab'][] = 'sort_by';

		if($data['group'] == ''){
			$data['group'] = 'group_by';

			$column_from_template_id = $this->report_builder_model->get_column_from_template_id($id);
			$data['column_of_table'] = $column_from_template_id['columns'];

			$column_group_by_from_template_id = $this->report_builder_model->get_column_group_by_from_template_id($id);
			$data['selected_columns'] = $column_group_by_from_template_id['group_by_columns'];
		}

		if($data['group'] == 'group_by'){
			$column_from_template_id = $this->report_builder_model->get_column_from_template_id($id);
			$data['column_of_table'] = $column_from_template_id['columns'];

			$column_group_by_from_template_id = $this->report_builder_model->get_column_group_by_from_template_id($id);
			$data['selected_columns'] = $column_group_by_from_template_id['group_by_columns'];

		}elseif($data['group'] == 'sort_by'){
			$get_all_column_from_template_id = $this->report_builder_model->get_all_column_from_template_id($id);
			$data['fields'] = $get_all_column_from_template_id['columns'];
			$data['sort_column_by_template_id'] = $this->report_builder_model->get_sort_column_by_template_id($id);
		}

		$data['rb_cell_type'] = rb_cell_type();

		$data['tabs']['view'] = 'templates/groups/'.$data['group'];

		$data['report_template_id'] = $id;
		$this->load->view('report_builder/templates/groups/group_manage', $data);
	}

	/**
	 * add group by columns
	 * @param string $id 
	 */
	public function add_group_by_columns($id='')
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('rb_groups');
		}

		if ($this->input->post()) {

			$data = $this->input->post();
			$result = $this->report_builder_model->add_group_by_columns($data, $id);

			if($result){
				set_alert('success', _l('rb_added_successfully'));
			}

			redirect(admin_url('report_builder/group_manage/'.$id.'?group=group_by'));
		}
	}

	/**
	 * add sort by columns
	 */
	public function add_sort_by_columns()
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('data_source');
		}

		if ($this->input->post()) {

			$data = $this->input->post();
			$templates_id = $data['templates_id'];

			$result = $this->report_builder_model->add_sort_by_column($data);

			if($result){
				set_alert('success', _l('rb_added_successfully'));
			}
			redirect(admin_url('report_builder/group_manage/'.$templates_id.'?group=sort_by'));
		}
	}

	/**
	 * subtotal manage
	 * @param  string $id 
	 * @return [type]     
	 */
	public function subtotal_manage($id = '')
	{
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('rb_subtotals');
		}

		$data=[];
		$data['title'] = _l('rb_subtotals');
		$data['header'] = _l('rb_subtotals');

		//get group by column
		$group_by_columns = [];
		$templates_selected_column = $this->report_builder_model->get_selected_column($id);
		$column_from_template_id = $templates_selected_column;
		foreach ($templates_selected_column as $selected_column) {
		    if($selected_column['group_by'] == 'yes'){
		    	$group_by_columns[] = $selected_column;
		    }
		}

		$data['templates_selected_column'] = $templates_selected_column;
		$data['group_by_columns'] = $group_by_columns;

		$subtotal_data = $this->report_builder_model->get_subtotal_data_from_template_id($id);
		$data['allow_subtotal'] = $subtotal_data['allow_subtotal'];
		$data['function_name'] = $subtotal_data['function_name'];
		$data['affected_columns'] = $subtotal_data['affected_columns'];

		$column_from_template_id = $this->report_builder_model->get_column_from_template_id($id, 'allow_subtotal');
		$data['columns'] = $column_from_template_id['columns'];

		$data['report_template_id'] = $id;

		$this->load->view('report_builder/templates/subtotals/subtotal_manage', $data);
	}

	/**
	 * add subtotals
	 * @param string $id 
	 */
	public function add_subtotals($id='')
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('rb_subtotals');
		}

		if ($this->input->post()) {

			$data = $this->input->post();
			$result = $this->report_builder_model->add_subtotals($data, $id);

			if($result){
				set_alert('success', _l('rb_added_successfully'));
			}

			redirect(admin_url('report_builder/subtotal_manage/'.$id));
		}
	}

	/**
	 * setting manage
	 * @param  string $id 
	 * @return [type]     
	 */
	public function template_setting_manage($id = '')
	{
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('rp_settings');
		}
		$this->load->model('staff_model');
		$this->load->model('departments_model');
		$this->load->model('roles_model');

		$data=[];
		$data['title'] = _l('rp_settings');
		$data['header'] = _l('rp_settings');

		$report_template = $this->report_builder_model->get_report_template($id);
		$data['report_template'] = $report_template;
		$data['general_setting_href']         = 'template_setting';
		$data['categories'] = $this->report_builder_model->get_category();
		$data['staffs'] = $this->staff_model->get();
		$data['departments'] = $this->departments_model->get();
		$data['roles']         = $this->roles_model->get();

		$data['report_template_id'] = $id;

		$this->load->view('report_builder/templates/template_settings/template_setting_manage', $data);
	}

	/**
	 * template setting
	 * @param  string $id 
	 * @return [type]     
	 */
	public function template_setting($id='')
	{
		if ($id == '') {
			blank_page('Report builder Not Found', 'danger');
		}

		if (!has_permission('report_builder', '', 'view') && !is_admin()) {
			access_denied('rp_settings');
		}

		$data = $this->input->post();

		if ($data) {
			$success = $this->report_builder_model->update_report_template($data, $id);

			if ($success == true) {
				$message = _l('rb_updated_successfully');
				set_alert('success', $message);
			}
			redirect(admin_url('report_builder/template_setting_manage/'.$id));
		}
	}

	/**
	 * add aggregation function
	 * @param string $id 
	 */
	public function add_aggregation_function($id='')
	{
	
		if (!has_permission('report_builder', '', 'create')  && !is_admin()) {
			access_denied('rb_subtotals');
		}

		if ($this->input->post()) {

			$data = $this->input->post();
			$result = $this->report_builder_model->add_aggregation_function($data, $id);

			if($result){
				set_alert('success', _l('rb_added_successfully'));
			}

			redirect(admin_url('report_builder/column_manage/'.$id.'?group=aggregation_function'));
		}
	}

	public function get_column_data()
	{
		$column_name = '';
		$table_name = $this->input->post('table_name');

		if($table_name != ''){
			$column_name = $this->report_builder_model->table_get_list_fields($table_name);
		}

		echo json_encode([
			'column_name' => $column_name,
		]);
	}

	/**
	 * get type of join data
	 * @return [type] 
	 */
	public function get_type_of_join_data()
	{
		$column_name = '';
		$data = $this->input->post();

		$result = $this->report_builder_model->relationship_get_table_type_of_join_data($data);

		echo json_encode([
			'related_table_data' => $result['related_table_data'],
			'type_of_join_data' => $result['type_of_join_data'],
			'related_table_data_html' => $result['related_table_data_html'],
			'array_type_of_join_data' => $result['array_type_of_join_data'],
		]);
	}

	/**
	 * get relationship row template
	 * @return [type] 
	 */
	public function get_relationship_row_template()
	{
		$data=[];
		

		$name = $this->input->post('name');
		$type_of_join = $this->input->post('type_of_join');
		$table_name = $this->input->post('table_name');
		$type_of_join_data_seleted = $this->input->post('type_of_join_data');
		$array_table_name = $this->input->post('array_table_name');
		$array_type_of_join_data = $this->input->post('this_array_type_of_join_data');
		$item_id = $this->input->post('item_id');

		$table_names=[];
		foreach ($array_table_name as $table_name) {
			$table_names[]=[
				'name' => $table_name,
				'label' => _l('tbl'.$table_name),
			];
		}

		$list_type_of_join_data=[];
		if($array_type_of_join_data){
			foreach ($array_type_of_join_data as $type_of_join_data) {
				$list_type_of_join_data[]=[
					'name' => $type_of_join_data,
				];
			}
		}


		echo new_html_entity_decode($this->report_builder_model->create_relationship_row_template($table_names, $list_type_of_join_data, $table_name, $type_of_join_data_seleted, $type_of_join, $item_id, $name, false));


	}

	/**
	 * template manage
	 * @return [type] 
	 */
	public function template_manage()
	{
	    if (!has_permission('report_builder', '', 'view') ) {
			access_denied('templates');
		}

		$data['title'] = _l('templates');
		$data['categories'] = $this->report_builder_model->get_category();

		$this->load->view('report_builder/templates/template_management', $data);
	}

	/**
	 * template table
	 * @return [type] 
	 */
	public function template_table()
	{
		$this->app->get_table_data(module_views_path('report_builder', 'templates/template_table'));
	}

	/**
	 * report detail
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function report_detail($id)
	{
		$search = [];
		if($this->input->post()){
			$filters = $this->input->post();

			foreach ($filters as $key => $value) {
				if(preg_match('/^filter_value_/', $key)){

					$key_explode = new_explode('#', $key);

						if(is_array($value)){

							$search[$key_explode[1]][$key_explode[0]] = implode(',', $value);
						}else{

							$search[$key_explode[1]][$key_explode[0]] = $value;
						}
				}
			}

		}

		$data=[];

		$report_group_by = false;
		$get_selected_column = $this->report_builder_model->get_selected_column($id);
		$get_subtotal = $this->report_builder_model->get_subtotal_data_from_template_id($id);
		//group by columns
		$group_by_columns = [];

		foreach ($get_selected_column as $value) {
			if($value['group_by'] == 'yes'){
				$report_group_by = true;
				$group_by_columns[] = db_prefix().$value['table_name'].'.'.$value['field_name'];

			}
		}

		$report_data = $this->report_builder_model->create_report_from_template($id, $search);
		$selected_column = $this->report_builder_model->get_selected_column($id);


		$data['cell_formattings'] = $this->report_builder_model->get_cell_formatting_by_template_id($id);
		$data['report_result'] = $report_data['rResult'];
		$data['report_columns'] = $selected_column;
		$data['report_group_by'] = $report_group_by;
		$data['id'] = $id;
		$data['search'] = $search;
		$data['group_by_columns'] = $group_by_columns;
		$data['get_selected_column'] = $get_selected_column;
		$data['report_template'] = $this->report_builder_model->get_report_template($id);
		$data['allow_subtotal'] = $get_subtotal['allow_subtotal'];

		$this->load->view('report_builder/reports/report_details/report_detail', $data);

	}

	/**
	 * report filter modal
	 * @return [type] 
	 */
	public function report_filter_modal()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}
		$data=[];
		$template_data=$this->input->post();
		$template_id=$template_data['template_id'];

		$filter_data = $this->report_builder_model->get_filter_data_by_template_id($template_id);

		$data['filter_data'] = $filter_data;
		$data['template_id'] = $template_id;

		$this->load->view('reports/report_details/report_filter_modal', $data);
	}

	/**
	 * delete template
	 * @param  [type] $template id 
	 * @return [type]              
	 */
	public function delete_template($id)
	{
	    if (!has_permission('report_builder', '', 'delete')  && !is_admin()) {
			access_denied('data_source');
		}

		$success = $this->report_builder_model->delete_template($id);
		if ($success) {
			set_alert('success', _l('rb_deleted'));
		} else {
			set_alert('warning', _l('problem_deleting'));
		}
		redirect(admin_url('report_builder/template_manage'));
	}

	/**
	 * get filter type
	 * @return [type] 
	 */
	public function get_filter_type()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}
		$rb_filter_type					= rb_filter_type();
		$rb_primary_foreign_key_field 	= rb_primary_foreign_key_field();
		$rb_number_field 				= rb_number_field();
		$rb_text_field 					= rb_text_field();
		$rb_date_field 					= rb_date_field();
		$rb_datetime_field 				= rb_datetime_field();

		$data = $this->input->get();
		$filter_type_options = '';
		$table_name = '';
		$field_name = '';
		$filter_type_selected = '';

		if(isset($data['cell_formatting'])){
			if($data['field_name']){
				$cell_formatting = new_explode('-', $data['field_name']);
				$table_name = $cell_formatting[1];
				$field_name = $cell_formatting[0];
				$filter_type_selected = $data['filter_type_selected'];
			}

		}else{
			$table_name = $data['table_name'];
			$field_name = $data['field_name'];
		}

		$filter_type_options = $this->report_builder_model->get_filter_type($table_name, $field_name, $filter_type_selected, true);

		

		echo json_encode([
			'filter_type' => $filter_type_options,
		]);
	}

	/**
	 * sharing table
	 * @return [type] 
	 */
	public function sharing_table()
	{
		$this->app->get_table_data(module_views_path('report_builder', 'sharings/sharing_table'));
	}

	/**
	 * sharing manage
	 * @return [type] 
	 */
	public function sharing_manage()
	{
	    if (!has_permission('report_builder', '', 'view') ) {
			access_denied('report');
		}

		$this->load->model('staff_model');
		$this->load->model('departments_model');
		$this->load->model('roles_model');

		$data['title'] = _l('sharings');
		$data['categories'] = $this->report_builder_model->get_category();
		$data['staffs'] = $this->staff_model->get();
		$data['departments'] = $this->departments_model->get();
		$data['roles']         = $this->roles_model->get();

		$this->load->view('report_builder/sharings/sharing_manage', $data);
	}

	/**
	 * sharing modal
	 * @return [type] 
	 */
	public function sharing_modal()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}
		$this->load->model('staff_model');
		$this->load->model('departments_model');
		$this->load->model('roles_model');

		$template_data=$this->input->post();
		$template_id=$template_data['template_id'];

		$data=[];
		$data['template_id'] = $template_id;

		if(isset($template_data['report'])){
			$data['report'] = true;
		}

		$data['get_report_setting'] = $this->report_builder_model->get_report_template($template_id);
		$data['staffs'] = $this->staff_model->get();
		$data['except_staffs'] = $this->staff_model->get('', 'staffid != '.get_staff_user_id());
		$data['departments'] = $this->departments_model->get();
		$data['roles']         = $this->roles_model->get();

		$this->load->view('report_builder/sharings/sharing_modal', $data);
	}

	/**
	 * report sharing setting
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function report_sharing_setting($template_id)
	{
		if (!has_permission('report_builder', '', 'view') ) {
			access_denied('report');
		}

		$report_manage = false;
		$data = $this->input->post();

		if(isset($data['report'])){
			$report_manage = true;
			unset($data['report']);
		}

		$success = $this->report_builder_model->update_report_template($data, $template_id);

		if ($success == true) {
			$message = _l('rb_updated_successfully');
			set_alert('success', $message);
		}
		if($report_manage){

			redirect(admin_url('report_builder/report_manage'));
		}else{

			redirect(admin_url('report_builder/sharing_manage'));
		}
	}

	/**
	 * get datasource filter value
	 * @return [type] 
	 */
	public function get_datasource_filter_value()
	{
		if (!$this->input->is_ajax_request()) {
			show_404();
		}

		$ajax_data = $this->input->get();
		$data = [];
		$filter_html = '';

		$rb_primary_foreign_key_field = rb_primary_foreign_key_field();
		$rb_number_field = rb_number_field();
		$rb_date_field = rb_date_field();
		$rb_datetime_field = rb_datetime_field();
		$data['rb_primary_foreign_key_field'] = $rb_primary_foreign_key_field;
		$data['rb_number_field'] = $rb_number_field;
		$data['rb_date_field'] = $rb_date_field;
		$data['rb_datetime_field'] = $rb_datetime_field;

		if(isset($ajax_data['cell_formatting'])){
			if($ajax_data['field_name']){

				$cell_formatting = new_explode('-', $ajax_data['field_name']);
				$table_name = $cell_formatting[1];
				$field_name = $cell_formatting[0];

				$data['filter_value'] = [
					'table_name' => $table_name,
					'field_name' => $field_name,
					'filter_type' => $ajax_data['filter_type'],
					'filter_value_1' => '',
					'filter_value_2' => '',
				];

			}

		}else{

			$data['filter_value'] = [
				'table_name' => $ajax_data['table_name'],
				'field_name' => $ajax_data['field_name'],
				'filter_type' => $ajax_data['filter_type'],
				'filter_value_1' => '',
				'filter_value_2' => '',
			];
		}

		$filter_html .= $this->load->view('report_builder/reports/report_details/render_input', $data, true);

		echo json_encode([
			'filter_value' => $filter_html,
		]);
	}


//end file
}