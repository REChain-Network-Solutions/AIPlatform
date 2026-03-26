<?php
defined('BASEPATH') or exit('No direct script access allowed');

class Report_builder_model extends App_Model
{
	public function __construct()
	{
		parent::__construct();
	}

	/**
	 * get data source realtionship by template_id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_data_source_relationship_by_template_id($template_id)
	{
		$this->db->where('templates_id', $template_id);
		return $this->db->get(db_prefix().'rb_data_source_relationships')->result_array();
	}

	/**
	 * get column realtionship by template_id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_column_relationship_by_template_id($template_id)
	{
		$this->db->where('templates_id', $template_id);
		return $this->db->get(db_prefix().'rb_columns')->result_array();
	}

	/**
	 * relationship modal get realated table
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function relationship_modal_get_realated_table($template_id)
	{
		$related_tables = rb_related_table();
		$ds_relationship = $this->get_data_source_relationship_by_template_id($template_id);

		if(count($ds_relationship) == 0){
			return $related_tables;
		}else{

			//old related table
			$old_related_table=[];
			$old_related_table_temp=[];
			$new_related_table=[];

			foreach ($ds_relationship as $value) {
				if(!in_array($value['left_table'], $old_related_table)){
					$old_related_table[] = $value['left_table'];
					$new_related_table[] = ['name' => $value['left_table'], 'label' => _l('tbl'.$value['left_table'])];
				}

			}

			//get new related table
			foreach ($related_tables as $related_table) {
				if(in_array($related_table['name'], $old_related_table)){

					foreach ($related_table['value'] as $table_name => $table_value) {
						if(!in_array($table_name, $old_related_table) && !in_array($table_name, $old_related_table_temp)){

							$new_related_table[] = ['name' => $table_name, 'label' => _l('tbl'.$table_name)];
							$old_related_table_temp[] = $table_name;
						}
					}
				}
			}
			return $new_related_table;
		}

	}

	/**
	 * get table related data
	 * @param  [type] $table 
	 * @return [type]        
	 */
	public function get_table_related_data($table)
	{
		$related_tables = rb_related_table();

		$left_field=[];
		$left_field_option_1='';
		$left_field_option_2='<option value=""></option>';
		$right_table='';

		$right_field_option_1='';
		$right_field_option_2='<option value=""></option>';

		foreach ($related_tables as $related_table) {
			if($related_table['name'] == $table){
				$related_data = $related_table['value'][$table];
				$left_field = array_merge($related_data['primary_key'], $related_data['foreign_key']);

				foreach ($left_field as $key => $left_field_value) {
					$left_field_option_1 .= '<option value="' . $left_field_value . '">' . $left_field_value . '</option>';
					if($key != 0){
						$left_field_option_2 .= '<option value="' . $left_field_value . '">' . $left_field_value . '</option>';
					}
				}

				foreach ($related_table['value'] as $table_name => $table_value) {
					if($table_name != $table){
						$right_table .= '<option value="' . $table_name . '">' . $table_name . '</option>';
						if(new_strlen($right_field_option_1) == 0){
							$table_related_data_v2 = $this->get_table_related_data_v2($table_name);

							$right_field_option_1 .= $table_related_data_v2['left_field_option_1'];
							$right_field_option_2 .= $table_related_data_v2['left_field_option_2'];
						}
					}
				}

				break;
			}
		}

		$results=[];
		$results['left_field_option_1'] = $left_field_option_1;
		$results['left_field_option_2'] = $left_field_option_2;
		$results['right_field_option_1'] = $right_field_option_1;
		$results['right_field_option_2'] = $right_field_option_2;
		$results['right_table'] = $right_table;

		return $results;
	}

	public function get_table_related_data_v2($table)
	{
		$related_tables = rb_related_table();

		$left_field=[];
		$left_field_option_1='<option value=""></option>';
		$left_field_option_2='<option value=""></option>';
		$right_table='';

		foreach ($related_tables as $related_table) {
			if($related_table['name'] == $table){
				$related_data = $related_table['value'][$table];
				$left_field = array_merge($related_data['primary_key'], $related_data['foreign_key']);

				foreach ($left_field as $key => $left_field_value) {
					$left_field_option_1 .= '<option value="' . $left_field_value . '">' . $left_field_value . '</option>';
					if($key != 0){
						$left_field_option_2 .= '<option value="' . $left_field_value . '">' . $left_field_value . '</option>';
					}
				}

				foreach ($related_table['value'] as $table_name => $table_value) {
					if($table_name != $table){
						$right_table .= '<option value="' . $table_name . '">' . $table_name . '</option>';
					}
				}

				break;
			}
		}

		$results=[];
		$results['left_field_option_1'] = $left_field_option_1;
		$results['left_field_option_2'] = $left_field_option_2;
		$results['right_table'] = $right_table;

		return $results;
	}

	/**
	 * add relationship
	 * @param [type] $data 
	 */
	public function add_relationship($data)
	{
		$affectedRows = 0;

		if(isset($data['join_type'])){
			unset($data['join_type']);
		}

		if(isset($data['left_table'])){
			unset($data['left_table']);
		}

		if(isset($data['query_string'])){
			unset($data['query_string']);
		}

		$newitems = [];
		if(isset($data['newitems'])){
			$newitems = $data['newitems'];
			unset($data['newitems']);
		}

		$items = [];
		if (isset($data['items'])) {
			$items = $data['items'];
			unset($data['items']);
		}

		$data['removed_items'] = isset($data['removed_items']) ? $data['removed_items'] : [];

		foreach ($data['removed_items'] as $remove_item_id) {
			if ($this->delete_relationship($remove_item_id)) {
				$affectedRows++;

			}
		}
		unset($data['removed_items']);

		//update
		if (count($items) > 0) {

			foreach ($items as $item) {
				$item['templates_id'] = $data['templates_id'];
				$this->db->where('id', $item['id']);
				$this->db->update(db_prefix() . 'rb_data_source_relationships', $item);
				if ($this->db->affected_rows() > 0) {
					$affectedRows++;
				}

			}
		}

		//insert
		if (count($newitems) > 0) {
			foreach ($newitems as $key => $newitem) {
				$newitem_insert = [];
				if(is_numeric($key)){

					$newitem_insert = [
						'join_type' => $newitem['join_type'],
						'left_table' => $newitem['left_table'],
						'query_string' => $newitem['query_string'],
						'templates_id' => $data['templates_id'],
					];
				}elseif($key != 'id'){
					$newitem_insert = [
						'left_table' => $newitem,
						'templates_id' => $data['templates_id'],

					];
				}

				if(count($newitem_insert) > 0){

					$this->db->insert(db_prefix() . 'rb_data_source_relationships', $newitem_insert);
					$insert_id = $this->db->insert_id();

					if($insert_id){
						$affectedRows++;
					}
				}
			}
		}

		if($affectedRows > 0){
			return true;
		}
		return false;
	}



	/**
	 * delete relationship
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_relationship($id)
	{	
		$affected_rows = 0;
		//delete data
		$this->db->where('id', $id);
		$this->db->delete(db_prefix() . 'rb_data_source_relationships');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		if($affected_rows > 0){
			return true;
		}
		return false;
	}

	/**
	 * get category
	 * @param  boolean $id 
	 * @return [type]      
	 */
	public function get_category($id = false)
	{

		if (is_numeric($id)) {
			$this->db->where('id', $id);

			return $this->db->get(db_prefix() . 'rb_categories')->row();
		}
		if ($id == false) {
			return $this->db->query('select * from '.db_prefix().'rb_categories')->result_array();
		}

	}

	/**
	 * add category
	 * @param [type] $data 
	 */
	public function add_category($data)
	{

		$this->db->insert(db_prefix() . 'rb_categories', $data);
		$insert_id = $this->db->insert_id();

		return $insert_id;
	}

	/**
	 * update category
	 * @param  [type] $data 
	 * @param  [type] $id   
	 * @return [type]       
	 */
	public function update_category($data, $id)
	{

		$this->db->where('id', $id);
		$this->db->update(db_prefix() . 'rb_categories', $data);

		if ($this->db->affected_rows() > 0) {
			return true;
		}

		return true;
	}

	/**
	 * delete category
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_category($id)
	{

		//delete series
		$this->db->where('id', $id);
		$this->db->delete(db_prefix() . 'rb_categories');

		if ($this->db->affected_rows() > 0) {
			return true;
		}

		return false;
	}

	/**
	 * get report setting
	 * @return [type] 
	 */
	public function get_report_setting()
	{
		return $this->db->get(db_prefix().'rb_settings')->row();
	}

	/**
	 * update report general setting
	 * @param  [type] $data 
	 * @return [type]       
	 */
	public function update_report_general_setting($data)
	{

		if(isset($data['is_public']) && $data['is_public'] == 'no'){
			if(!isset($data['role_id'])){
				$data['role_id'] = null;
			}else{
				$data['role_id'] = implode(",", $data['role_id']);
			}

			if(!isset($data['department_id'])){
				$data['department_id'] = null;
			}else{
				$data['department_id'] = implode(",", $data['department_id']);
			}

			if(!isset($data['staff_id'])){
				$data['staff_id'] = null;
			}else{
				$data['staff_id'] = implode(",", $data['staff_id']);
			}

		}else{
			$data['role_id'] = null;
			$data['department_id'] = null;
			$data['staff_id'] = null;
		}
		//check insert or update
		$report_setting = $this->get_report_setting();
		if($report_setting){
			$this->db->where('id', $report_setting->id);
			$this->db->update(db_prefix().'rb_settings', $data);
			if ($this->db->affected_rows() > 0) {
				return true;
			}else{
				return false;
			}
		}else{
			$this->db->insert(db_prefix().'rb_settings', $data);
			$insert_id = $this->db->insert_id();

			if($insert_id){
				return true;
			}else{
				return false;
			}
		}

	}

	/**
	 * add report setting
	 */
	public function add_report_setting()
	{
		$report_setting = $this->get_report_setting();
		if(isset($report_setting)){
			$data = get_object_vars($report_setting);

			if(isset($data['id'])){
				unset($data['id']);
			}
			$data['staff_create'] = get_staff_user_id();
			$data['date_create'] = date('Y-m-d H:i:s');

			$this->db->insert(db_prefix().'rb_templates', $data);
			$insert_id = $this->db->insert_id();
			if($insert_id){
				return $insert_id;
			}else{
				return false;
			}

		}else{
			return 'need_add_general_setting_before_create_report';
		}
	}

	/**
	 * filter modal get table
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function filter_modal_get_table($template_id)
	{
		$tables=[];
		$temp_table=[];

		$ds_relationship = $this->get_data_source_relationship_by_template_id($template_id);

		foreach ($ds_relationship as $value) {
			if(!in_array($value['left_table'], $temp_table)){
				$temp_table[] = $value['left_table'];
				$tables[] = ['name' => $value['left_table'], 'label' => _l('tbl'.$value['left_table'])];
			}

			if($value['right_table'] != '' && !in_array($value['right_table'], $temp_table)){
				$temp_table[] = $value['right_table'];
				$tables[] = ['name' => $value['right_table'], 'label' => _l('tbl'.$value['right_table'])];
			}
		}

		return $tables;
	}

	/**
	 * table get list fields
	 * @param  [type] $table 
	 * @return [type]        
	 */
	public function table_get_list_fields($table, $array = false)
	{	

		$list_fields = $this->db->list_fields($table);

		if($array){
			$fileds_option=[];
			foreach ($list_fields as $key => $value) {
				$fileds_option[] = [
					'id' => $value ,
					'description' =>  _l('tbl'.$table.'_'.$value) ,
				];

			}

		}else{
			$fileds_option='';
			$fileds_option .= '<option value=""></option>';

			foreach ($list_fields as $key => $value) {
				$fileds_option .= '<option value="' . $value . '">' . _l('tbl'.$table.'_'.$value) . '</option>';
			}
		}
		return $fileds_option;
	}

	/**
	 * add filter
	 * @param [type] $data 
	 */
	public function add_filter($data, $id)
	{

		if(isset($data['ask_user'])){
			if($data['ask_user'] == 'yes'){
				$data['ask_user'] = 'yes';
			}else{
				$data['ask_user'] = 'no';
			}
		}else{
			$data['ask_user'] = 'no';
		}

		if(is_numeric($id)){
			$this->db->where('id', $id);
			$this->db->update(db_prefix() . 'rb_data_source_filters', $data);

			if ($this->db->affected_rows() > 0) {
				return true;
			}

			return true;
		}else{

			$this->db->insert(db_prefix().'rb_data_source_filters', $data);
			$insert_id = $this->db->insert_id();

			if($insert_id){
				return $insert_id;
			}
			
			return false;
		}

	}

	/**
	 * delete filter
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_filter($id)
	{	
		$affected_rows = 0;
		$this->db->where('id', $id);
		$this->db->delete(db_prefix() . 'rb_data_source_filters');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		if($affected_rows > 0){
			return true;
		}
		return false;
	}

	/**
	 * get next step report
	 * @param  [type] $template_id 
	 * @param  [type] $type        
	 * @return [type]              
	 */
	public function get_next_step_report($template_id, $type)
	{
		$status = false;
		$next_link ='';
		switch ($type) {
			case 'data_source':
				$data_sources = $this->get_data_source_relationship_by_template_id($template_id);
				if(count($data_sources) > 0){
					$status = true;
					$next_link .= admin_url('report_builder/group_manage/'.$template_id.'?group=group_by');
				}

				break;

			case 'group':
				$status = true;
				$next_link .= admin_url('report_builder/subtotal_manage/'.$template_id);

				break;

			case 'subtotal':
				$status = true;

				$column_group_by_from_template = $this->get_column_group_by_from_template_id($template_id);
				$data['group_by_columns'] = $column_group_by_from_template['group_by_columns'];

				$next_link .= admin_url('report_builder/column_manage/'.$template_id.'?group=column');

				break;

			case 'column':
				$data_sources = $this->get_column_relationship_by_template_id($template_id);
				if(count($data_sources) > 0){
					$status = true;
					$next_link .= admin_url('report_builder/cell_formatting/'.$template_id);
				}
				break;

			case 'cell_formatting':
				$status = true;
				$next_link .= admin_url('report_builder/template_setting_manage/'.$template_id);

				break;

			case 'setting':

				//handle create report from template
				$report_data = $this->create_report_from_template($template_id);
				$status = true;

				$next_link .= admin_url('report_builder/report_manage');

				break;
			
		}

		$data=[];
		$data['status'] = $status;
		$data['next_link'] = $next_link;

		return $data;
	}

	/**
	 * get selected column
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_selected_column($templates_id)
	{
		$this->db->where('templates_id', $templates_id);
		$this->db->order_by('order_display', 'ASC');
		$columns = $this->db->get(db_prefix().'rb_columns')->result_array();
		return $columns;
	}

	/**
	 * concat column with table
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function concat_column_with_table($template_id)
	{
		$selected_columns=[];
		$selected_columns_temp=[];
		$selected_columns_data=[];

		$get_selected_column = $this->get_selected_column($template_id);

		foreach ($get_selected_column as $value) {
			$selected_columns[] = [
				'name' => $value['table_name'].'/'.$value['field_name'],
				'label' => _l('tbl'.$value['table_name'].'_'.$value['field_name']).' ('._l('tbl'.$value['table_name']).')',
			];
			$selected_columns_temp[]= $value['table_name'].'/'.$value['field_name'];
			$selected_columns_data[$value['table_name'].'/'.$value['field_name']] = $value;

		}

		$data=[];
		$data['selected_columns'] =  $selected_columns;
		$data['selected_columns_temp'] =  $selected_columns_temp;
		$data['selected_columns_data'] =  $selected_columns_data;

		return $data;
	}

	/**
	 * get column from template id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_column_from_template_id($template_id, $group_subtotal='group_by')
	{	
		$columns=[];
		$tables=[];
		$ds_relationship = $this->get_data_source_relationship_by_template_id($template_id);
		$concat_column_with_table = $this->concat_column_with_table($template_id);
		$selected_columns = $concat_column_with_table['selected_columns'];
		$selected_columns_temp = $concat_column_with_table['selected_columns_temp'];
		$selected_columns_data = $concat_column_with_table['selected_columns_data'];

		foreach ($ds_relationship as $relationship) {
			if(!in_array($relationship['left_table'], $tables)){
				$tables[] = $relationship['left_table'];
			}
			if(!in_array($relationship['right_table'], $tables)){
				$tables[] = $relationship['right_table'];
			}
		}

		foreach ($tables as $table) {
			if($table != ''){
				$list_fields = $this->db->list_fields($table);
				foreach ($list_fields as $list_field) {
					

					if(!in_array($table.'/'.$list_field, $selected_columns_temp) ){

						$columns[] = [
							'name' => $table.'/'.$list_field,
							'label' => _l('tbl'.$table.'_'.$list_field).' ('._l('tbl'.$table).')',
							

						];

					}elseif(in_array($table.'/'.$list_field, $selected_columns_temp)){
						if($group_subtotal == 'group_by'){

							if($selected_columns_data[$table.'/'.$list_field]['group_by'] == 'no'){
								$columns[] = [
									'name' => $table.'/'.$list_field,
									'label' => _l('tbl'.$table.'_'.$list_field).' ('._l('tbl'.$table).')',
								];
							}
						}else{
							//sub total
							if($selected_columns_data[$table.'/'.$list_field]['affected_column'] == 'no'){
																$columns[] = [
									'name' => $table.'/'.$list_field,
									'label' => _l('tbl'.$table.'_'.$list_field).' ('._l('tbl'.$table).')',
								];
							}
						}
					}

				}
			}
		}

		$data=[];
		$data['selected_columns'] = $selected_columns;
		$data['columns'] = $columns;

		return $data;
	}

	/**
	 * add columns
	 * @param [type] $data 
	 */
	public function add_columns($data, $template_id)
	{
		$affectedRows=0;

		if(isset($data['to'])){
			$column_insert=[];
			$column_delete=[];
			$column_update=[];

			//check each column => add or update or delete
			$concat_column_with_table = $this->concat_column_with_table($template_id);
			$old_columns = $concat_column_with_table['selected_columns_temp'];

			foreach ($data['to'] as $key => $table_with_column) {
				if(in_array($table_with_column, $old_columns)){
					//update
					//
					$index = array_search($table_with_column, $old_columns);
					unset($old_columns[$index]);
				}else{
					//insert
					$column_insert[] =[
						'templates_id' => $template_id,
						'table_name' => new_explode('/', $table_with_column)[0],
						'field_name' => new_explode('/', $table_with_column)[1],
						'label_name' => ucfirst(new_str_replace('_', ' ', new_explode('/', $table_with_column)[1])),
					];

				}
			}

			//insert batch
			if(count($column_insert) > 0){
				$affected_rows = $this->db->insert_batch(db_prefix().'rb_columns', $column_insert);
				if($affected_rows > 0){
					$affectedRows++;
				}
			}

			//delete
			if(count($old_columns) > 0){
				$table_name='';
				$field_name='';

				foreach ($old_columns as $old_column) {
					$table_column = new_explode('/', $old_column);

					if(new_strlen($table_name) > 0){
						$table_name .= ','.'"'.$table_column[0].'"';
					}else{
						$table_name .= '"'.$table_column[0].'"';
					}

					if(new_strlen($field_name) > 0){
						$field_name .= ','.'"'.$table_column[1].'"';
					}else{
						$field_name .= '"'.$table_column[1].'"';
					}
					
				}

				$this->db->where('templates_id', $template_id);
				$this->db->where('table_name IN ('. $table_name.') ');
				$this->db->where('field_name IN ('. $field_name.') ');
				$this->db->delete(db_prefix().'rb_columns');
				if($this->db->affected_rows() > 0){
					$affectedRows++;
				}
			}

		}else{
			$this->db->where('templates_id', $template_id);
			$this->db->where('group_by', 'no');
			$this->db->delete(db_prefix().'rb_columns');
			if($this->db->affected_rows() > 0){
				$affectedRows++;
			}
		}

		if($affectedRows){
			return true;
		}
		return false;
	}

	/**
	 * add label cell type
	 * @param [type] $data 
	 */
	public function add_label_cell_type($data)
	{
		$affectedRows = 0;

		if (isset($data['label_cell_type_hs'])) {
			$label_cell_type_hs = $data['label_cell_type_hs'];
			unset($data['label_cell_type_hs']);
		}

		if(isset($label_cell_type_hs)){
			$label_cell_type_detail = json_decode($label_cell_type_hs);

			$es_detail = [];
			$row = [];
			$rq_val = [];
			$header = [];

			$header[] = 'order_display';
			$header[] = 'table_name';
			$header[] = 'field_name';
			$header[] = 'label_name';
			$header[] = 'field_type';
			$header[] = 'templates_id';
			$header[] = 'id';

			foreach ($label_cell_type_detail as $key => $value) {
				//only get row "value" != 0
				if($value[5] != ''){
					$es_detail[] = array_combine($header, $value);
				}
			}
		}
		$row = [];
		$row['update'] = []; 
		$row['insert'] = []; 
		$row['delete'] = [];
		$total = [];

		foreach ($es_detail as $key => $value) {
			if($value['id'] != ''){
				unset($value['table_name']);
				unset($value['field_name']);
				unset($value['templates_id']);
				
				$row['delete'][] = $value['id'];
				$row['update'][] = $value;
			}else{
				unset($value['id']);
				$row['insert'][] = $value;
			}

		}

		if(count($row['update']) != 0){
			$affected_rows = $this->db->update_batch(db_prefix().'rb_columns', $row['update'], 'id');
			if($affected_rows > 0){
				$affectedRows++;
			}
		}

		if ($affectedRows > 0) {
			return true;
		}

		return false;
	}

	
	/**
	 * Adds a cell formatting.
	 *
	 * @param        $data   The data
	 * @param        $id     The identifier
	 */
	public function add_cell_formatting($data){

		$fields = new_explode('-', $data['field_name']);

		$data['table_name'] = $fields[1];
		$data['field_name'] = $fields[0];

		$this->db->insert(db_prefix().'rb_field_conditional_formattings', $data);
		$insert_id = $this->db->insert_id();
		if($insert_id){
			return $insert_id;
		}
		return false;
	}

	 /**
	 * delete relationship
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_cell_formatting($id)
	{	
		$affected_rows = 0;
		//delete data
		$this->db->where('id', $id);
		$this->db->delete(db_prefix() . 'rb_field_conditional_formattings');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		if($affected_rows > 0){
			return true;
		}
		return false;
	}

	/**
	 * cell formatting
	 * @param  [type] $data 
	 * @param  [type] $id   
	 * @return [type]       
	 */
	public function update_cell_formatting($data, $id)
	{	
		$fields = new_explode('-', $data['field_name']);

		$data['table_name'] = $fields[1];
		$data['field_name'] = $fields[0];

		$this->db->where('id', $id);
		$this->db->update(db_prefix() . 'rb_field_conditional_formattings', $data);

		if ($this->db->affected_rows() > 0) {
			return true;
		}

		return true;
	}

	/**
	 * get cell formatting
	 * @param  boolean $id 
	 * @return [type]      
	 */
	public function get_cell_formatting($id = false)
	{

		if (is_numeric($id)) {
			$this->db->where('id', $id);

			return $this->db->get(db_prefix() . 'rb_field_conditional_formattings')->row();
		}
		if ($id == false) {
			return $this->db->query('select * from '.db_prefix().'rb_field_conditional_formattings')->result_array();
		}

	}

	/**
	 * get cell formatting by template_id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_cell_formatting_by_template_id($template_id)
	{

		$this->db->where('templates_id', $template_id);
		$conditional_formattings = $this->db->get(db_prefix() . 'rb_field_conditional_formattings')->result_array();
		$data = [];
		foreach ($conditional_formattings as $conditional_formatting) {
			if(isset($data[db_prefix().$conditional_formatting['table_name'].'.'.$conditional_formatting['field_name']])){
				$data[db_prefix().$conditional_formatting['table_name'].'.'.$conditional_formatting['field_name']]['filter_values'][] = $conditional_formatting;
			}else{
				$data[db_prefix().$conditional_formatting['table_name'].'.'.$conditional_formatting['field_name']]['filter_type'] = $conditional_formatting['filter_type'];
				$data[db_prefix().$conditional_formatting['table_name'].'.'.$conditional_formatting['field_name']]['filter_values'][] = $conditional_formatting;
			}
		}

		return $data;

	}


	/**
	 * get column group by from template id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_column_group_by_from_template_id($template_id)
	{	

		$non_group_by_columns=[];
		$group_by_columns=[];

		$get_selected_column = $this->get_selected_column($template_id);

		foreach ($get_selected_column as $value) {
			if($value['group_by'] == 'yes'){
				$group_by_columns[] = [
					'name' => $value['id'],
					'label' => _l('tbl'.$value['table_name'].'_'.$value['field_name']).' ('._l('tbl'.$value['table_name']).')',

				];
			}else{
				$non_group_by_columns[] = [
					'name' => $value['id'],
					'label' => _l('tbl'.$value['table_name'].'_'.$value['field_name']).' ('._l('tbl'.$value['table_name']).')',
					
				];
			}

		}

		$data=[];
		$data['group_by_columns'] =  $group_by_columns;
		$data['non_group_by_columns'] =  $non_group_by_columns;

		return $data;

	}

	/**
	 * add group by columns
	 * @param [type] $data        
	 * @param [type] $template_id 
	 */
	public function add_group_by_columns($data, $template_id)
	{
		$selected_columns = [];
		$get_selected_column = $this->get_selected_column($template_id);
		foreach ($get_selected_column as $value) {
		    $selected_columns[$value['table_name'].'/'.$value['field_name']] = $value;
		}

		$affectedRows=0;
		if(isset($data['from'])){
			if(count($data['from']) > 0){
				$update_no=[];
				foreach ($data['from'] as $value) {
					$update_no[] = [
						'id' 		=> $value,
						'group_by' 	=> 'no',
					];
				}

				if(count($update_no) > 0){
					$affected_rows = $this->db->update_batch(db_prefix().'rb_columns', $update_no, 'id');
					if($affected_rows > 0){
						$affectedRows++;
					}
				}
			}
		}

		if(isset($data['to'])){
			if(count($data['to']) > 0){

				$insert_yes=[];
				$update_yes=[];
				foreach ($data['to'] as $value) {

					//update if value is numeric
					if(is_numeric($value)){
						$update_yes[] = [
							'id' 		=> $value,
							'group_by' 	=> 'yes',
						];
					}else{
					//insert if value is string
					//check if exist => update else => insert

						if(isset($selected_columns[$value])){
						//update
							$update_yes[] = [
								'id' 		=> $selected_columns[$value]['id'],
								'group_by' 	=> 'yes',
							];

						}else{
						//insert
							$insert_yes[] =[
								'templates_id' => $template_id,
								'table_name' => new_explode('/', $value)[0],
								'field_name' => new_explode('/', $value)[1],
								'label_name' => new_str_replace('_', ' ', new_explode('/', $value)[1]),
								'group_by' 	=> 'yes',

							];
						}
					}
				}

				if(count($update_yes) > 0){
					$affected_rows = $this->db->update_batch(db_prefix().'rb_columns', $update_yes, 'id');
					if($affected_rows > 0){
						$affectedRows++;
					}
				}

				if(count($insert_yes) > 0){
					$affected_rows = $this->db->insert_batch(db_prefix().'rb_columns', $insert_yes);
					if($affected_rows > 0){
						$affectedRows++;
					}
				}
			}
		}

		if($affectedRows){
			return true;
		}
		return false;
	}

	/**
	 * get all column from template id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_all_column_from_template_id($template_id)
	{	
		$group_by_columns=[];
		$get_selected_column = $this->get_selected_column($template_id);
		foreach ($get_selected_column as $value) {
			if($value['group_by'] == 'yes'){
				$group_by_columns[] = $value['table_name'].'_'.$value['field_name']; 
			}
		}
		
		$columns=[];
		$tables=[];
		$ds_relationship = $this->get_data_source_relationship_by_template_id($template_id);
		$concat_column_with_table = $this->concat_column_with_table($template_id);
		$selected_columns = $concat_column_with_table['selected_columns'];
		$selected_columns_temp = $concat_column_with_table['selected_columns_temp'];

		foreach ($ds_relationship as $relationship) {
			if(!in_array($relationship['left_table'], $tables)){
				$tables[] = $relationship['left_table'];
			}
			if(!in_array($relationship['right_table'], $tables)){
				$tables[] = $relationship['right_table'];
			}
		}

		foreach ($tables as $table) {

			if($table != ''){
				$list_fields = $this->db->list_fields($table);
				foreach ($list_fields as $list_field) {
					if(count($group_by_columns) > 0){
						if(in_array($table.'_'.$list_field, $group_by_columns)){
							$columns[] = [
								'label' => $list_field.'-'.$table,
								'field_name' => _l('tbl'.$table.'_'.$list_field).' ('._l('tbl'.$table).')',
								'table_name' => _l('tbl'.$table),
							];
						}
					}else{

						$columns[] = [
							'label' => $list_field.'-'.$table,
							'field_name' => _l('tbl'.$table.'_'.$list_field).' ('._l('tbl'.$table).')',
							'table_name' => _l('tbl'.$table),
						];
					}
				}
			}
		}

		$data=[];
		$data['columns'] = $columns;

		return $data;
	}

	/**
	 * get sort column
	 * @param  boolean $id 
	 * @return [type]      
	 */
	public function get_sort_column_by_template_id($template_id = false)
	{

		$this->db->where('templates_id', $template_id);
		return $this->db->get(db_prefix() . 'rb_sort_bys')->result_array();
		
	}

	/**
	 * add sort by column
	 * @param [type] $data 
	 */
	public function add_sort_by_column($data)
	{

		$templates_id 	= $data['templates_id'];
		$ids 			= $data['id'];
		$field_names 	= $data['field_name'];
		if(isset($data['order_by'])){
			$order_bys =  $data['order_by'];
		}

		$affectedRows 		= 0;
		$column_sort_insert = [];
		$column_sort_delete = [];
		$column_sort_update = [];
		$list_ids 			= [];

		foreach ($data['field_name'] as $key => $field_name) {
			if($field_name != ''){

				$fields = new_explode('-', $field_name);
				$table_name = $fields[1];
				$field_name = $fields[0];

				if($ids[$key] == 0){
					//insert
					if(isset($order_bys)){
						if($order_bys[$key] == 'on' ){
							$column_sort_insert[] = [
								'templates_id'	=> $templates_id, 
								'table_name' 	=> $table_name, 
								'field_name' 	=> $field_name, 
								'order_by' 		=> 'DESC', 
							];
						}else{
							$column_sort_insert[] = [
								'templates_id'	=> $templates_id, 
								'table_name' 	=> $table_name, 
								'field_name' 	=> $field_name, 
								'order_by' 		=> 'ASC', 
							];
						}
					}else{
						$column_sort_insert[] = [
							'templates_id'	=> $templates_id, 
							'table_name' 	=> $table_name, 
							'field_name' 	=> $field_name, 
							'order_by' 		=> 'ASC', 
						];
					}

				}else{
					$list_ids[] = $ids[$key];

					//update
					if(isset($order_bys)){
						if($order_bys[$key] == 'on' ){
							$column_sort_update[] = [
								'id'			=> $ids[$key], 
								'templates_id'	=> $templates_id, 
								'table_name' 	=> $table_name, 
								'field_name' 	=> $field_name, 
								'order_by' 		=> 'DESC', 
							];
						}else{
							$column_sort_update[] = [
								'id'			=> $ids[$key], 
								'templates_id'	=> $templates_id, 
								'table_name' 	=> $table_name, 
								'field_name' 	=> $field_name, 
								'order_by' 		=> 'ASC', 
							];
						}
					}else{
						$column_sort_update[] = [
							'id'			=> $ids[$key], 
							'templates_id'	=> $templates_id, 
							'table_name' 	=> $table_name, 
							'field_name' 	=> $field_name, 
							'order_by' 		=> 'ASC', 
						];
					}

				}

			}
		}

		//delete
		if(count($list_ids) > 0){

			$this->db->where('templates_id', $templates_id);
			$this->db->where('id NOT IN ('. implode(",", $list_ids) .') ');
			$this->db->delete(db_prefix().'rb_sort_bys');
			if($this->db->affected_rows() > 0){
				$affectedRows++;
			}

		}else{

			$this->db->where('templates_id', $templates_id);
			$this->db->delete(db_prefix().'rb_sort_bys');
			if($this->db->affected_rows() > 0){
				$affectedRows++;
			}
		}

		//insert
		if(count($column_sort_insert) > 0){
			$affected_rows = $this->db->insert_batch(db_prefix().'rb_sort_bys', $column_sort_insert);
			if($affected_rows > 0){
				$affectedRows++;
			}
		}

		//update
		if(count($column_sort_update) > 0){
			$affected_rows = $this->db->update_batch(db_prefix().'rb_sort_bys', $column_sort_update, 'id');
			if($affected_rows > 0){
				$affectedRows++;
			}
		}

		if($affectedRows > 0){
			return true;
		}

		return false;
	}


	/**
	 * get subtotal from template id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_subtotal_data_from_template_id($template_id)
	{	
		$allow_subtotal = '';
		$function_name 	= '';

		$columns=[];
		$affected_columns=[];

		$get_selected_column = $this->get_selected_column($template_id);

		foreach ($get_selected_column as $value) {
			
			if($value['allow_subtotal'] == 'yes'){
				$allow_subtotal = $value['allow_subtotal'];
			}

			if($value['function_name'] != null){
				$function_name = $value['function_name'];
			}

			if($value['affected_column'] == 'yes'){
				$affected_columns[] = [
					'name' => $value['id'],
					'label' => _l('tbl'.$value['table_name'].'_'.$value['field_name']).' ('._l('tbl'.$value['table_name']).')',
				];
			}else{
				$columns[] = [
					'name' => $value['id'],
					'label' => _l('tbl'.$value['table_name'].'_'.$value['field_name']).' ('._l('tbl'.$value['table_name']).')',
				];
			}

		}

		if($function_name == ''){
			$function_name = 'sum';
		}

		$data=[];
		$data['allow_subtotal'] 	=  $allow_subtotal;
		$data['function_name'] 		=  $function_name;
		$data['affected_columns'] 	=  $affected_columns;
		$data['columns'] 			=  $columns;

		return $data;

	}

	/**
	 * add subtotals
	 * @param [type] $data        
	 * @param [type] $template_id 
	 */
	public function add_subtotals($data, $template_id)
	{
		$selected_columns = [];
		$get_selected_column = $this->get_selected_column($template_id);
		foreach ($get_selected_column as $value) {
			$selected_columns[$value['table_name'].'/'.$value['field_name']] = $value;
		}

		$affectedRows=0;
		$allow_subtotal = 'no';
		$function_name = $data['function_name'];

		if(isset($data['allow_subtotal'])){
			if($data['allow_subtotal'] == 'on'){
				$allow_subtotal = 'yes';
			}
		}


		if(isset($data['from'])){
			if(count($data['from']) > 0){
				$update_no=[];
				foreach ($data['from'] as $value) {
					$update_no[] = [
						'id' 				=> $value,
						'function_name' 	=> $function_name,
						'allow_subtotal' 	=> 'no',
						'affected_column' 	=> 'no',
					];
				}

				if(count($update_no) > 0){
					$affected_rows = $this->db->update_batch(db_prefix().'rb_columns', $update_no, 'id');
					if($affected_rows > 0){
						$affectedRows++;
					}
				}
			}
		}

		if(isset($data['to'])){
			if(count($data['to']) > 0){
				$insert_yes=[];
				$update_yes=[];
				foreach ($data['to'] as $value) {
					if(is_numeric($value)){

						$update_yes[] = [
							'id' 				=> $value,
							'function_name' 	=> $function_name,
							'allow_subtotal' 	=> $allow_subtotal,
							'affected_column' 	=> 'yes',
						];
					}else{
						if(isset($selected_columns[$value])){
						//update
							$update_yes[] = [
								'id' 		=> $selected_columns[$value]['id'],
								'affected_column' 	=> 'yes',
							];

						}else{
						//insert
					//insert if value is string
							$insert_yes[] =[
								'templates_id' => $template_id,
								'table_name' => new_explode('/', $value)[0],
								'field_name' => new_explode('/', $value)[1],
								'label_name' => new_str_replace('_', ' ', new_explode('/', $value)[1]),
								'function_name' 	=> $function_name,
								'allow_subtotal' 	=> $allow_subtotal,
								'affected_column' 	=> 'yes',

							];
						}
					}


				}

				if(count($update_yes) > 0){
					$affected_rows = $this->db->update_batch(db_prefix().'rb_columns', $update_yes, 'id');
					if($affected_rows > 0){
						$affectedRows++;
					}
				}

				if(count($insert_yes) > 0){
					$affected_rows = $this->db->insert_batch(db_prefix().'rb_columns', $insert_yes);
					if($affected_rows > 0){
						$affectedRows++;
					}
				}
			}
		}

		if($affectedRows){
			return true;
		}
		return false;
	}

	/**
	 * get report setting
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function get_report_template($id)
	{
		if (is_numeric($id)) {
			$this->db->where('id', $id);

			return $this->db->get(db_prefix() . 'rb_templates')->row();
		}
		if ($id == false) {
			return $this->db->query('select * from '.db_prefix().'rb_templates')->result_array();
		}
	}

	/**
	 * update report template
	 * @param  [type] $data 
	 * @param  [type] $id   
	 * @return [type]       
	 */
	public function update_report_template($data, $id)
	{

		if(isset($data['is_public']) && $data['is_public'] == 'no'){
			if(!isset($data['role_id'])){
				$data['role_id'] = null;
			}else{
				$data['role_id'] = implode(",", $data['role_id']);
			}

			if(!isset($data['department_id'])){
				$data['department_id'] = null;
			}else{
				$data['department_id'] = implode(",", $data['department_id']);
			}

			if(!isset($data['staff_id'])){
				$data['staff_id'] = null;
			}else{
				$data['staff_id'] = implode(",", $data['staff_id']);
			}

		}else{
			$data['role_id'] = null;
			$data['department_id'] = null;
			$data['staff_id'] = null;
		}

		if(isset($data['except_staff'])){
			$data['except_staff'] = implode(",", $data['except_staff']);

		}else{
			$data['except_staff'] = null;

		}

		//check insert or update
		$report_setting = $this->get_report_setting();
		$this->db->where('id', $id);
		$this->db->update(db_prefix().'rb_templates', $data);
		if ($this->db->affected_rows() > 0) {
			return true;
		}else{
			return false;
		}

	}

	/**
	 * get aggregation function
	 * @param  [type] $templates_id 
	 * @return [type]               
	 */
	public function get_aggregation_function($templates_id)
	{
		$this->db->where('templates_id', $templates_id);
		$aggregation_function = $this->db->get(db_prefix().'rb_aggregation_functions')->row();
		return $aggregation_function;
	}

	/**
	 * add aggregation function
	 * @param [type] $data 
	 */
	public function add_aggregation_function($data, $templates_id)
	{

		$id = $data['id'];
		unset($data['id']);
		
		$allow_aggregation_function = 'no';

		if(isset($data['allow_aggregation_function'])){
			if($data['allow_aggregation_function'] == 'on'){
				$allow_aggregation_function = 'yes';
			}
		}

		$data['allow_aggregation_function'] = $allow_aggregation_function;
		$data['templates_id'] = $templates_id;

		if(is_numeric($id)){
			//update
			$this->db->where('id', $id);
			$this->db->update(db_prefix() . 'rb_aggregation_functions', $data);

			if ($this->db->affected_rows() > 0) {
				return true;
			}

		}else{
			//insert
			$this->db->insert(db_prefix().'rb_aggregation_functions', $data);
			if($insert_id){
				return $insert_id;
			}

		}
	}

	/**
	 * create relationship row template
	 * @param  [type]  $tables_data     
	 * @param  string  $table_seleted   
	 * @param  string  $column_selected 
	 * @param  string  $type            
	 * @param  boolean $main_table      
	 * @return [type]                   
	 */
	public function create_relationship_row_template($tables_data, $list_type_of_join_data=[], $table_seleted = '', $type_of_join_data_seleted = '', $type_of_join = '', $r_id = '', $name = '',  $main_table = false, $required = true)
	{

				
		$table_label_name = '';
		$type_of_join_label_name = '';
		$join_string_label_name = '';
		if($main_table){
			$table_label_name = 'rb_table';
			$type_of_join_label_name = 'rb_type_of_join';
			$join_string_label_name = 'rb_type_of_join_data';
		}

		$row = '';
		$list_column = [];
		$name_left_table = 'left_table';
		$query_string = 'query_string';
		$name_join_type = 'join_type';


		
		if ($name == '') {

			$row .= '<div class="row main relationship_row">';

		} else {
			$row .= '<div class=" row sortable item relationship_row">
					<input type="hidden" class="ids" name="' . $name . '[id]" value="' . $r_id . '">';

			$name_left_table = $name . '[left_table]';
			$query_string = $name . '[query_string]';
			$name_join_type = $name . '[join_type]';

		}

		$select_attrs=[];
		if($required){
			$select_attrs = ['required' => true];
		}

		//use for delivery note
		$array_table_attr = ["onchange" => "table_onchange('" . $name_left_table . "','" . $name_join_type . "','" . $query_string . "','" . $main_table . "');"];
		if($required){
			$array_table_attr = ["onchange" => "table_onchange('" . $name_left_table . "','" . $name_join_type . "','" . $query_string . "','" . $main_table . "');", "required" => true];
		}
		
		if($main_table == false){
			$row .= '<div class="col-md-3 type_of_join">' . render_select($name_join_type, rb_join_type(), array('name', 'label'), '', $type_of_join, $select_attrs, [], '') . '</div>';
		}else{
			$row .= '<div class="col-md-3 type_of_join"><div class="form-group"><label>'._l($type_of_join_label_name).'</label></div></div>';
		}

		$row .= '<div class="col-md-4 table_name">' . render_select($name_left_table, $tables_data, array('name', array('label')), $table_label_name, $table_seleted, $array_table_attr, [], '') . '</div>';

		if($main_table == false){
			$row .= '<div class="col-md-4 type_of_join_data">' . render_select($query_string, $list_type_of_join_data, array('name', array('name')), $join_string_label_name, $type_of_join_data_seleted, $select_attrs, [], '') . '</div>';
		}else{
			$row .= '<div class="col-md-4 type_of_join_data"><div class="form-group"><label>'._l($join_string_label_name).'</label></div></div>';
		}


		
		if ($name == '') {
			$row .= '<div class="col-md-1"><button type="button" onclick="add_relationship_item_to_table(\'undefined\',\'undefined\'); return false;" class="btn pull-right btn-info"><i class="fa fa-check"></i></button></div>';
		} else {
			if($main_table == false){
				$row .= '<div class="col-md-1"><button type="button" class="btn pull-right btn-danger " onclick="delete_relationship_item(this,' . $r_id . ',\'.relationship_data\'); return false;"><i class="fa fa-trash"></i></button></div>';
			}
		}

		$row .= '</div>';
		return $row;
	}

	/**
	 * get filter data by template id
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function get_filter_data_by_template_id($template_id)
	{
		$this->db->where('templates_id', $template_id);
		return $this->db->get(db_prefix().'rb_data_source_filters')->result_array();
	}

	/**
	 * get datasource filter
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function get_datasource_filter($id)
	{
		$this->db->where('id', $id);
		return $this->db->get(db_prefix().'rb_data_source_filters')->row();
	}

	/**
	 * create_where_condition
	 * @param  [type] $filter_value 
	 * @return [type]               
	 */
	public function create_condition_default($filter_value)
	{
		$where = '';
		switch($filter_value['filter_type']) {
			case 'equal':
			if(!is_numeric($filter_value['filter_value_1'])){
				$filter_value['filter_value_1'] = '"'.$filter_value['filter_value_1'].'"';
			}
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' = '.$filter_value['filter_value_1'];
			break;

			case 'greater_than':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' > '.$filter_value['filter_value_1'];
			break;

			case 'less_than':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' < '.$filter_value['filter_value_1'];
			break;

			case 'greater_than_or_equal':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' >= '.$filter_value['filter_value_1'];
			break;

			case 'less_than_or_equal':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' <= '.$filter_value['filter_value_1'];
			break;

			case 'between':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' BETWEEN "'.$filter_value['filter_value_1'].'" AND "'. $filter_value['filter_value_2'].'"';
			break;

			case 'like':
			if(substr($filter_value['filter_value_1'], -1) == '"'){
				$filter_value['filter_value_1'] = substr($filter_value['filter_value_1'], 0,  -1).'%"';
			}
			if(substr($filter_value['filter_value_1'], 0, 1) == '"'){
				$filter_value['filter_value_1'] = '"%'.substr($filter_value['filter_value_1'], 1);
			}

			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' LIKE '.$filter_value['filter_value_1'];
			break;

			case 'NOT_like':
			if(substr($filter_value['filter_value_1'], -1) == '"'){
				$filter_value['filter_value_1'] = substr($filter_value['filter_value_1'], 0,  -1).'%"';
			}
			if(substr($filter_value['filter_value_1'], 0, 1) == '"'){
				$filter_value['filter_value_1'] = '"%'.substr($filter_value['filter_value_1'], 1);
			}
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' NOT LIKE '.$filter_value['filter_value_1'];
			break;

			case 'not_equal':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' != '.$filter_value['filter_value_1'];
			break;

			case 'begin_with':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' LIKE '.$filter_value['filter_value_1'].'%';
			break;

			case 'end_with':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' LIKE %'.$filter_value['filter_value_1'];
			break;

			case 'in':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' IN ('.$filter_value['filter_value_1'].')';
			break;

			case 'not_in':
			$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' NOT IN ('.$filter_value['filter_value_1'].')';
			break;
		}

		return $where;
	}


	public function create_condition_ask_user($filter_value, $search)
	{
		$where = '';

		switch($filter_value['filter_type']) {
			case 'equal':
			if(isset($search[$filter_value['id']]['filter_value_1']) && !is_numeric($search[$filter_value['id']]['filter_value_1'])){
				$search[$filter_value['id']]['filter_value_1'] = '"'.$search[$filter_value['id']]['filter_value_1'].'"';
			}
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' = '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'greater_than':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' > '.$search[$filter_value['id']]['filter_value_1'];
			}
			break;

			case 'less_than':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' < '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'greater_than_or_equal':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' >= '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'less_than_or_equal':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' <= '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'between':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' BETWEEN "'.$search[$filter_value['id']]['filter_value_1'].'" AND "'. $search[$filter_value['id']]['filter_value_2'].'"';
			}

			break;

			case 'like':
			if(isset($search[$filter_value['id']])){
				if(substr($search[$filter_value['id']]['filter_value_1'], -1) == '"'){
					$search[$filter_value['id']]['filter_value_1'] = substr($search[$filter_value['id']]['filter_value_1'], 0,  -1).'%"';
				}
				if(substr($search[$filter_value['id']]['filter_value_1'], 0, 1) == '"'){
					$search[$filter_value['id']]['filter_value_1'] = '"%'.substr($search[$filter_value['id']]['filter_value_1'], 1);
				}
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' LIKE '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'NOT_like':
			if(isset($search[$filter_value['id']])){
				if(substr($search[$filter_value['id']]['filter_value_1'], -1) == '"'){
					$search[$filter_value['id']]['filter_value_1'] = substr($search[$filter_value['id']]['filter_value_1'], 0,  -1).'%"';
				}
				if(substr($search[$filter_value['id']]['filter_value_1'], 0, 1) == '"'){
					$search[$filter_value['id']]['filter_value_1'] = '"%'.substr($search[$filter_value['id']]['filter_value_1'], 1);
				}
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' NOT LIKE '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'not_equal':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' != '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'begin_with':
			if(isset($search[$filter_value['id']])){

				if(substr($search[$filter_value['id']]['filter_value_1'], -1) == '"'){
					$search[$filter_value['id']]['filter_value_1'] = substr($search[$filter_value['id']]['filter_value_1'], 0,  -1).'%"';

				// //last: example
				// 	substr($search[$filter_value['id']]['filter_value_1'], -1);
				// 	substr('a,b,c,d,e,', 0, -1);

					//first: example
				// 	substr($search[$filter_value['id']]['filter_value_1'], 0, 1);
				// 	$str1 = substr($str, 1);

				}

				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' LIKE '.$search[$filter_value['id']]['filter_value_1'];
			}
			break;

			case 'end_with':
			if(isset($search[$filter_value['id']])){

				if(substr($search[$filter_value['id']]['filter_value_1'], 0, 1) == '"'){
					$search[$filter_value['id']]['filter_value_1'] = '"%'.substr($search[$filter_value['id']]['filter_value_1'], 1);
				}

				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' LIKE '.$search[$filter_value['id']]['filter_value_1'];
			}

			break;

			case 'in':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' IN ('.$search[$filter_value['id']]['filter_value_1'].')';
			}
			break;

			case 'not_in':
			if(isset($search[$filter_value['id']])){
				$where = $filter_value['group_condition'].' '.db_prefix().$filter_value['table_name'].'.'.$filter_value['field_name'].' NOT IN ('.$search[$filter_value['id']]['filter_value_1'].')';
			}
			break;
		}

		return $where;
	}

	/**
	 * create children report from template
	 * @param  [type] $template_id 
	 * @param  array  $search      
	 * @return [type]              
	 */
	public function create_children_report_from_template($template_id, $search = [], $filter_by_parent = [])
	{		
		//table	
		$main_table = '';	
		$join_table = [];
		$relationship_data = $this->get_data_source_relationship_by_template_id($template_id);
		$get_selected_column = $this->get_selected_column($template_id);


		foreach ($relationship_data as $key => $value) {
			if($value['join_type'] == '' || $value['join_type'] ==  null){
				$main_table = db_prefix().$value['left_table'];
			}else{
				if($value['join_type'] == 'inner_join'){

					$join_table[] = 'INNER JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];
				}elseif($value['join_type'] == 'left_join'){

					$join_table[] = 'LEFT JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];

				}elseif($value['join_type'] == 'right_join'){
					$join_table[] = 'RIGHT JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];

				}elseif($value['join_type'] == 'full_outer_join'){
					$join_table[] = 'FULL OUTER JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];

				}

			}

			
		}

		//group by columns
		$group_by_columns = [];


		//filter
		//NOTE: If in WHERE Clause have SUM() , MAx, MIN, COUNT, AVG => Having clause 
		$where=[];
		$filter_data = $this->get_filter_data_by_template_id($template_id);
		foreach ($filter_data as $filter_value) {
			if($filter_value['ask_user'] == 'no'){
				if(count(new_explode(',', $filter_value['filter_value_1'])) > 1){
					$filter_value1_str = '';
					$filter_value_temp = new_explode(',', $filter_value['filter_value_1']);
					foreach ($filter_value_temp as $key => $value) {
					    if(new_strlen($filter_value1_str) > 0){
							$filter_value1_str .= ',"'.$value.'"';
						}else{
							$filter_value1_str .= '"'.$value.'"';
						}
					}
					$filter_value['filter_value_1'] = $filter_value1_str;
				}

				$where[] = $this->create_condition_default($filter_value);

			}else{
				if(count(new_explode(',', $filter_value['filter_value_1'])) > 1){
					$filter_value1_str = '';
					$filter_value_temp = new_explode(',', $filter_value['filter_value_1']);
					foreach ($filter_value_temp as $key => $value) {
					    if(new_strlen($filter_value1_str) > 0){
							$filter_value1_str .= ',"'.$value.'"';
						}else{
							$filter_value1_str .= '"'.$value.'"';
						}
					}
					$filter_value['filter_value_1'] = $filter_value1_str;
				}
				$where[] = $this->create_condition_ask_user($filter_value, $search);

			}
		    
		}

		if(count($filter_by_parent) > 0){
			$where = array_merge($where, $filter_by_parent);
		}

		//seleted column
		//Must check if have group by column => only slect column in Group by + SUM, MIN, MAX, AVG
		$selected_columns=[];
		foreach ($get_selected_column as $key => $value) {
			if(count($group_by_columns) > 0){

				$column_name = db_prefix().$value['table_name'].'.'.$value['field_name'];
				if(in_array($column_name, $group_by_columns)){
					$selected_columns[] =  $column_name;
				}

			}else{
				$selected_columns[] =  db_prefix().$value['table_name'].'.'.$value['field_name'];
			}
		}

		$selected_columns = array_values($selected_columns);

		//cell formatting
		$cell_formattings = $this->get_cell_formatting($template_id);


		//order by column
		$orderby=[];
		$sort_by_columns = $this->get_sort_column_by_template_id($template_id);
		foreach ($sort_by_columns as $sort_value) {
		    $orderby[] = db_prefix().$sort_value['table_name'].'.'.$sort_value['field_name'].' '.$sort_value['order_by'];
		}
		
		//get sub total by column, if have group by => will have sum, count, max, min, avg
		$additional_select = [];
		if(count($group_by_columns) > 0){

			foreach ($get_selected_column as $value) {
				if($value['affected_column'] == 'yes' && $value['allow_subtotal'] == 'yes'){

					$additional_select[] = $value['function_name'].'('.db_prefix().$value['table_name'].'.'.$value['field_name'].') AS `'.$value['function_name'].'_'.db_prefix().$value['table_name'].'.'.$value['field_name'].'`';

				}
			}
		}

		//aggregation functions
		$get_aggregation_function = $this->get_aggregation_function($template_id);

		if($get_aggregation_function){
			if($get_aggregation_function->allow_aggregation_function == 'yes'){
				if(new_strlen($get_aggregation_function->affected_column) > 0){

				}	

			}
		}


		$result_data = $this->sql_query_init($selected_columns, 'id', $main_table , $join_table , $where, $additional_select, $group_by_columns, [], $orderby);

		return $result_data;

	}


	/**
	 * create report from template
	 * @param  [type] $template_id 
	 * @return [type]              
	 */
	public function create_report_from_template($template_id, $search = [])
	{		
		//table	
		$main_table = '';	
		$join_table = [];
		$relationship_data = $this->get_data_source_relationship_by_template_id($template_id);
		$get_selected_column = $this->get_selected_column($template_id);


		foreach ($relationship_data as $key => $value) {
			if($value['join_type'] == '' || $value['join_type'] ==  null){
				$main_table = db_prefix().$value['left_table'];
			}else{
				if($value['join_type'] == 'inner_join'){

					$join_table[] = 'INNER JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];
				}elseif($value['join_type'] == 'left_join'){

					$join_table[] = 'LEFT JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];

				}elseif($value['join_type'] == 'right_join'){
					$join_table[] = 'RIGHT JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];

				}elseif($value['join_type'] == 'full_outer_join'){
					$join_table[] = 'FULL OUTER JOIN ' . db_prefix() . $value['left_table'].' ON ' . $value['query_string'];

				}

			}

			
		}

		//group by columns
		$group_by_columns = [];

		foreach ($get_selected_column as $value) {
			if($value['group_by'] == 'yes'){

				$group_by_columns[] = db_prefix().$value['table_name'].'.'.$value['field_name'];
			}
		}


		//filter
		//NOTE: If in WHERE Clause have SUM() , MAx, MIN, COUNT, AVG => Having clause 
		$where=[];
		$filter_data = $this->get_filter_data_by_template_id($template_id);
		foreach ($filter_data as $filter_value) {
			if($filter_value['ask_user'] == 'no'){
				if(count(new_explode(',', $filter_value['filter_value_1'])) > 1){
					$filter_value1_str = '';
					$filter_value_temp = new_explode(',', $filter_value['filter_value_1']);
					foreach ($filter_value_temp as $key => $value) {
						if(new_strlen($filter_value1_str) > 0){
							$filter_value1_str .= ',"'.$value.'"';
						}else{
							$filter_value1_str .= '"'.$value.'"';
						}
					}
					$filter_value['filter_value_1'] = $filter_value1_str;
				}
				$where[] = $this->create_condition_default($filter_value);

			}else{
				if(count(new_explode(',', $filter_value['filter_value_1'])) > 1){
					$filter_value1_str = '';
					$filter_value_temp = new_explode(',', $filter_value['filter_value_1']);
					foreach ($filter_value_temp as $key => $value) {
					    if(new_strlen($filter_value1_str) > 0){
							$filter_value1_str .= ',"'.$value.'"';
						}else{
							$filter_value1_str .= '"'.$value.'"';
						}
					}
					$filter_value['filter_value_1'] = $filter_value1_str;
				}
				$where[] = $this->create_condition_ask_user($filter_value, $search);

			}

		}



		//seleted column
		//Must check if have group by column => only slect column in Group by + SUM, MIN, MAX, AVG
		$selected_columns=[];
		foreach ($get_selected_column as $key => $value) {
			if(count($group_by_columns) > 0){

				$column_name = db_prefix().$value['table_name'].'.'.$value['field_name'];
				if(in_array($column_name, $group_by_columns)){
					$selected_columns[] =  $column_name;
				}

			}else{
				$selected_columns[] =  db_prefix().$value['table_name'].'.'.$value['field_name'];
			}
		}

		$selected_columns = array_values($selected_columns);

		//cell formatting
		$cell_formattings = $this->get_cell_formatting($template_id);


		//order by column
		$orderby=[];
		$sort_by_columns = $this->get_sort_column_by_template_id($template_id);
		foreach ($sort_by_columns as $sort_value) {
		    $orderby[] = db_prefix().$sort_value['table_name'].'.'.$sort_value['field_name'].' '.$sort_value['order_by'];
		}
		
		//get sub total by column, if have group by => will have sum, count, max, min, avg
		$additional_select = [];
		if(count($group_by_columns) > 0){

			foreach ($get_selected_column as $value) {
				if($value['affected_column'] == 'yes' && $value['allow_subtotal'] == 'yes'){

					$additional_select[] = $value['function_name'].'('.db_prefix().$value['table_name'].'.'.$value['field_name'].') AS `'.$value['function_name'].'_'.db_prefix().$value['table_name'].'.'.$value['field_name'].'`';

				}
			}
		}

		//aggregation functions
		$get_aggregation_function = $this->get_aggregation_function($template_id);

		$affected_column = '';
		if($get_aggregation_function){
			if($get_aggregation_function->allow_aggregation_function == 'yes'){
				if(new_strlen($get_aggregation_function->affected_column) > 0){
					$affected_column = new_explode(',', $get_aggregation_function->affected_column);
				}	

			}
		}
		
		//limit
		$limit_data = 0;
		$get_report_template = $this->get_report_template($template_id);
		if($get_report_template){
			$limit_data = (int)$get_report_template->records_per_page;
		}

		$result_data = $this->sql_query_init($selected_columns, 'id', $main_table , $join_table , $where, $additional_select, $group_by_columns, [], $orderby, [], $limit_data);

		return $result_data;

	}

	/**
	 * sql query init
	 * @param  [type] $aColumns         
	 * @param  [type] $sIndexColumn     
	 * @param  [type] $sTable           
	 * @param  array  $join             
	 * @param  array  $where            
	 * @param  array  $additionalSelect 
	 * @param  string $sGroupBy         
	 * @param  array  $searchAs         
	 * @return [type]                   
	 */
	function sql_query_init($aColumns, $sIndexColumn, $sTable, $join = [], $where = [], $additionalSelect = [], $GroupBy = [], $having = [],  $orderby = [], $searchAs = [], $limit=0)
	{
		$havingCount = '';
		
		$sLimit = '';
		if($limit > 0){
			$sLimit = 'LIMIT '.$limit;
		}
		
		//column selected
		$_aColumns = [];
		foreach ($aColumns as $column) {
			// if found only one dot
			if (substr_count($column, '.') == 1 && strpos($column, ' as ') === false) {
				$_column = new_explode('.', $column);
				if (isset($_column[1])) {
					if (startsWith($_column[0], db_prefix())) {
						$_prefix = prefixed_table_fields_wildcard($_column[0], $_column[0], $_column[1]);
						array_push($_aColumns, $_prefix);
					} else {
						array_push($_aColumns, $column);
					}
				} else {
					array_push($_aColumns, $_column[0]);
				}
			} else {
				array_push($_aColumns, $column);
			}
		}

		/*
		 * group by
		 */

		$sGroupBy = '';

		if(count($GroupBy) > 0){
			$group_value = array_values($GroupBy);
			$sGroupBy .= 'GROUP BY '. implode(",", $group_value);
		}

		/*
		*sHaving
		*/
	
		$sHaving = '';

		if(count($having) > 0){
			$having_value = array_values($having);
			$sHaving .= 'HAVING '. implode(",", $having_value);
		}

		/*
		 * Ordering by
		 */

		$sOrder = '';

		if(count($orderby) > 0){
			$order_value = array_values($orderby);
			$sOrder .= 'ORDER BY '. implode(",", $order_value);
		}


		/*
		 * Filtering
		 */
		$sWhere = '';



		/*
		 * SQL queries
		 * Get data to display
		 */
		$_additionalSelect = '';
		if (count($additionalSelect) > 0) {
			$_additionalSelect = ',' . implode(',', $additionalSelect);
		}

		$where = implode(' ', $where);

		if ($sWhere == '') {
			$where = trim($where);
			if (startsWith($where, 'AND') || startsWith($where, 'OR')) {
				if (startsWith($where, 'OR')) {
					$where = substr($where, 2);
				} else {
					$where = substr($where, 3);
				}
				$where = 'WHERE ' . $where;
			}
		}

		$join = implode(' ', $join);
		$sQuery = '
		SELECT SQL_CALC_FOUND_ROWS ' . new_str_replace(' , ', ' ', implode(', ', $_aColumns)) . ' ' . $_additionalSelect . "
		FROM $sTable
		" . $join . "
		$sWhere
		" . $where . "
		$sGroupBy
		$sHaving
		$sOrder
		$sLimit
		";

		if(new_strlen($sQuery) > 81 && (count($_aColumns) > 0 || new_strlen($_additionalSelect) > 0)){
			$rResult = $this->db->query($sQuery)->result_array();
		}else{
			$rResult = [];
		}


		/* Data set length after filtering */
		$sQuery = '
		SELECT FOUND_ROWS()
		';

		if(new_strlen($sQuery) > 27){
			$_query         = $this->db->query($sQuery)->result_array();
		}else{
			$_query = [];
		}

		if(count($_query) > 0){
			$iFilteredTotal = $_query[0]['FOUND_ROWS()'];
		}else{
			$iFilteredTotal = [];
		}
		if (startsWith($where, 'AND')) {
			$where = 'WHERE ' . substr($where, 3);
		}
		
		$output = [
			'iTotalDisplayRecords' => $iFilteredTotal,
			'aaData'               => [],
		];

		return [
			'rResult' => $rResult,
			'output'  => $output,
		];
	}

	/**
	 * relationship get table type of join data
	 * @param  [type] $data 
	 * @return [type]       
	 */
	public function relationship_get_table_type_of_join_data($data)
	{
		$related_table_data = [];
		$related_table_data_html = '';
		$type_of_join_data = '';
		$array_type_of_join_data = [];
		$select_data_type_of_join = [];

		$rb_related_table = rb_related_table();
		$array_table_names = $data['array_table_names'];
		$related_table_data_temp = [];
		$table_names = $data['table_names'];
		$rb_related_table_temp = [];

		foreach ($rb_related_table as $related_table) {

			if( isset($data['main_table']) && is_numeric($data['main_table']) && $data['table_names'] == ''){

				$related_table_data = rb_related_table();
				$related_table_data_html .= '<option value="' . $related_table['name'] . '">' . $related_table['label'] . '</option>'; 

			}else{
				if(in_array($related_table['name'], $array_table_names)){

					foreach ($related_table['value'] as $table_name => $table_value) {
						if(!in_array($table_name, $related_table_data_temp)){

							$related_table_data[] = ['name' => $table_name, 'label' => _l('tbl'.$table_name)];
							$related_table_data_temp[] = $table_name;
							$related_table_data_html .= '<option value="' . $table_name . '">' . _l('tbl'.$table_name) . '</option>'; 
						}

						if($table_name == $table_names && $related_table['name'] != $table_names){
							if(isset($table_value['operator_str'])){

								if(is_array($table_value['operator_str'])){

									foreach ($table_value['operator_str'] as $operator_str_value) {
										if(!in_array($operator_str_value, $array_type_of_join_data)){
											$type_of_join_data .= '<option value="' . $operator_str_value . '">' . $operator_str_value . '</option>'; 
											$array_type_of_join_data[] = $operator_str_value;
											$select_data_type_of_join[] = ['name' => $operator_str_value];
										}

									}

								}else{
									$type_of_join_data .= '<option value="' . $table_value['operator_str'] . '">' . $table_value['operator_str'] . '</option>'; 
									$array_type_of_join_data[] = $table_value['operator_str'];
									$select_data_type_of_join[] = ['name' => $table_value['operator_str']];
								}


							}
						}
					}
				}
			}


		}

		$result_data = [];
		$result_data['related_table_data'] = $related_table_data;
		$result_data['type_of_join_data'] = $type_of_join_data;
		$result_data['related_table_data_html'] = $related_table_data_html;
		$result_data['array_type_of_join_data'] = $array_type_of_join_data;
		$result_data['select_data_type_of_join'] = $select_data_type_of_join;

		return $result_data;
	}

	/**
	 * delete template
	 * @param  [type] $id 
	 * @return [type]     
	 */
	public function delete_template($id)
	{	

		$affected_rows = 0;
		//delete template
		$this->db->where('id', $id);
		$this->db->delete(db_prefix() . 'rb_templates');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		//delete template
		$this->db->where('templates_id', $id);
		$this->db->delete(db_prefix() . 'rb_data_source_relationships');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		//delete template
		$this->db->where('templates_id', $id);
		$this->db->delete(db_prefix() . 'rb_data_source_filters');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		//delete rb_columns
		$this->db->where('templates_id', $id);
		$this->db->delete(db_prefix() . 'rb_columns');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}


		//delete rb_columns
		$this->db->where('templates_id', $id);
		$this->db->delete(db_prefix() . 'rb_aggregation_functions');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		//delete rb_columns
		$this->db->where('templates_id', $id);
		$this->db->delete(db_prefix() . 'rb_sort_bys');
		if ($this->db->affected_rows() > 0) {
			$affected_rows++;
		}

		if($affected_rows > 0){
			return true;
		}
		return false;
	}

	public function payslip_template_get_staffid($department_ids, $role_ids, $staff_ids, $except_staff='')
	{
		if(new_strlen($staff_ids) > 0){
			if( new_strlen($except_staff) > 0){
				$array_except_staff = explode(",", $except_staff);
				$array_staff_ids = explode(",", $except_staff);

				$new_staff_ids=[];
				foreach ($array_staff_ids as $value) {
				    if(!in_array($value, $array_except_staff)){
				    	$new_staff_ids[] = $value;
				    }
				}

				if(count($new_staff_ids) > 0){
					return implode(",", $new_staff_ids);
				}
				return '';
			}else{
				return $staff_ids;
			}
		}	
		
		$department_querystring='';
		$role_querystring='';
		$except_staff_querystring='';

		if(new_strlen($department_ids) > 0){
			$arrdepartment = $this->staff_model->get('', 'staffid in (select '.db_prefix().'staff_departments.staffid from '.db_prefix().'staff_departments where departmentid IN( '.$department_ids.'))');
			$temp = '';
			foreach ($arrdepartment as $value) {
				$temp = $temp.$value['staffid'].',';
			}
			$temp = rtrim($temp,",");
			$department_querystring = 'FIND_IN_SET(staffid, "'.$temp.'")';
		}

		if( new_strlen($role_ids) > 0){
			$role_querystring = 'FIND_IN_SET(role, "'.$role_ids.'")';
		}

		if( new_strlen($except_staff) > 0){
			$except_staff_querystring = 'staffid NOT IN ('.$except_staff .')' ;
		}

		$arrQuery = array($department_querystring, $role_querystring, $except_staff_querystring);

		$newquerystring = '';
		foreach ($arrQuery as $string) {
			if($string != ''){
				$newquerystring = $newquerystring.$string.' AND ';
			}            
		}  

		$newquerystring=rtrim($newquerystring,"AND ");
		if($newquerystring == ''){
			$newquerystring = [];
		}
		$staffs = $this->get_staff_timekeeping_applicable_object($newquerystring);
		$staff_ids=[];
		foreach ($staffs as $key => $value) {
		    $staff_ids[] = $value['staffid'];
		}

		if(count($staff_ids) > 0){
			return implode(',', $staff_ids);
		}
		return false;
	}
	
	/**
	 * get filter type
	 * @param  [type] $table_name           
	 * @param  [type] $field_name           
	 * @param  [type] $filter_type_selected 
	 * @param  [type] $html                 
	 * @return [type]                       
	 */
	public function get_filter_type($table_name, $field_name, $filter_type_selected, $html)
	{
		$rb_filter_type					= rb_filter_type();
		$rb_primary_foreign_key_field 	= rb_primary_foreign_key_field();
		$rb_number_field 				= rb_number_field();
		$rb_text_field 					= rb_text_field();
		$rb_date_field 					= rb_date_field();
		$rb_datetime_field 				= rb_datetime_field();
		$filter_type_options			= '';
		$array_filter_type_options		= [];

		if($table_name != '' && $field_name != '' ){
			$filter_type_options .= '<option value=""></option>';

			if(isset($rb_primary_foreign_key_field['tbl'.$table_name.'_'.$field_name])){

				foreach ($rb_filter_type as $filter_type_value) {
					if($filter_type_value['name'] == 'equal' || $filter_type_value['name'] == 'in' || $filter_type_value['name'] == 'not_in' ){
						$selected = '';
						if($filter_type_selected == $filter_type_value['name']){
							$selected = ' selected';
						}

						$filter_type_options .= '<option value="' . $filter_type_value['name'] . '" '.$selected.'>' . $filter_type_value['label'] . '</option>';

						$array_filter_type_options[] = [
							'name' => $filter_type_value['name'],
							'label' => $filter_type_value['label'],
						];

					}
				}

			}elseif(isset($rb_number_field['tbl'.$table_name.'_'.$field_name])){
				foreach ($rb_filter_type as $filter_type_value) {
					if($filter_type_value['name'] == 'equal' || $filter_type_value['name'] == 'greater_than' || $filter_type_value['name'] == 'less_than' || $filter_type_value['name'] == 'greater_than_or_equal' || $filter_type_value['name'] == 'less_than_or_equal' || $filter_type_value['name'] == 'between' || $filter_type_value['name'] == 'not_equal' ){

						$selected = '';
						if($filter_type_selected == $filter_type_value['name']){
							$selected = ' selected';
						}

						$filter_type_options .= '<option value="' . $filter_type_value['name'] . '" '.$selected.'>' . $filter_type_value['label'] . '</option>';
						$array_filter_type_options[] = [
							'name' => $filter_type_value['name'],
							'label' => $filter_type_value['label'],
						];
					}
				}

			}elseif(isset($rb_date_field['tbl'.$table_name.'_'.$field_name])){
				foreach ($rb_filter_type as $filter_type_value) {
					if($filter_type_value['name'] == 'between' ){
						$selected = '';
						if($filter_type_selected == $filter_type_value['name']){
							$selected = ' selected';
						}

						$filter_type_options .= '<option value="' . $filter_type_value['name'] . '" '.$selected.'>' . $filter_type_value['label'] . '</option>';
						$array_filter_type_options[] = [
							'name' => $filter_type_value['name'],
							'label' => $filter_type_value['label'],
						];
					}
				}

			}elseif(isset($rb_datetime_field['tbl'.$table_name.'_'.$field_name])){
				foreach ($rb_filter_type as $filter_type_value) {
					if($filter_type_value['name'] == 'between' ){
						$selected = '';
						if($filter_type_selected == $filter_type_value['name']){
							$selected = ' selected';
						}

						$filter_type_options .= '<option value="' . $filter_type_value['name'] . '" '.$selected.'>' . $filter_type_value['label'] . '</option>';
						$array_filter_type_options[] = [
							'name' => $filter_type_value['name'],
							'label' => $filter_type_value['label'],
						];
					}
				}

			}else{
				foreach ($rb_filter_type as $filter_type_value) {
					if($filter_type_value['name'] == 'equal' || $filter_type_value['name'] == 'like' || $filter_type_value['name'] == 'NOT_like' || $filter_type_value['name'] == 'begin_with' || $filter_type_value['name'] == 'end_with' || $filter_type_value['name'] == 'in' || $filter_type_value['name'] == 'not_in' ){
						$selected = '';
						if($filter_type_selected == $filter_type_value['name']){
							$selected = ' selected';
						}

						$filter_type_options .= '<option value="' . $filter_type_value['name'] . '" '.$selected.'>' . $filter_type_value['label'] . '</option>';
						$array_filter_type_options[] = [
							'name' => $filter_type_value['name'],
							'label' => $filter_type_value['label'],
						];
					}
				}

			}

		}

		if($html){
			return $filter_type_options;
		}else{
			return $array_filter_type_options;
		}

	}



//end file
}