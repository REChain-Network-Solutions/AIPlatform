<?php

defined('BASEPATH') or exit('No direct script access allowed');
$this->ci->load->model('departments_model');
$this->ci->load->model('staff_model');

$aColumns = [
	db_prefix() . 'rb_templates.id',
	'report_title',
	'category_id',
	'staff_create',
	'date_create',
	'1',
];
$sIndexColumn = 'id';
$sTable = db_prefix() . 'rb_templates';

$where = [];
$join = [
	'LEFT JOIN ' . db_prefix() . 'rb_categories ON ' . db_prefix() . 'rb_templates.category_id = ' . db_prefix() . 'rb_categories.id',
];

$category_filter = $this->ci->input->post('category_filter');
$staff_filter = $this->ci->input->post('staff_filter');
if(isset($category_filter)){
	$where_category = '';
	foreach ($category_filter as $category_id) {

		if($category_id != '')
		{
			if($where_category == ''){
				$where_category .= ' ('.db_prefix().'rb_templates.category_id in ('.$category_id.')';
			}else{
				$where_category .= ' or '.db_prefix().'rb_templates.category_id in ('.$category_id.')';
			}
		}
	}
	if($where_category != '')
	{
		$where_category .= ')';
		if($where != ''){
			array_push($where, 'AND'. $where_category);
		}else{
			array_push($where, $where_category);
		}
		
	}
}

if(isset($staff_filter)){
	$where_staff_create = '';
	foreach ($staff_filter as $staff_id) {

		if($staff_id != '')
		{
			if($where_staff_create == ''){
				$where_staff_create .= ' ('.db_prefix().'rb_templates.staff_create in ('.$staff_id.')';
			}else{
				$where_staff_create .= ' or '.db_prefix().'rb_templates.staff_create in ('.$staff_id.')';
			}
		}
	}
	if($where_staff_create != '')
	{
		$where_staff_create .= ')';
		if($where != ''){
			array_push($where, 'AND'. $where_staff_create);
		}else{
			array_push($where, $where_staff_create);
		}
		
	}
}


$staff_user_id = get_staff_user_id();
if(!is_admin()){

	//get_staff_departments
	$staff_departments = $this->ci->departments_model->get_staff_departments($staff_user_id, true);
	$staff_infor = $this->ci->staff_model->get($staff_user_id);
	$role_id = 0;
	if($staff_infor){
		$role_id = $staff_infor->role;
	}

}

$result = data_tables_init($aColumns, $sIndexColumn, $sTable, $join, $where, [db_prefix() . 'rb_templates.id', 'name', 'except_staff', 'role_id', 'department_id', 'staff_id', 'is_public']);

$output = $result['output'];
$rResult = $result['rResult'];

foreach ($rResult as $aRow) {
	$row = [];

	//report shrare with me + owner. Admin view all
	if(!is_admin()){
		$arr_except_staff = new_explode(',', $aRow['except_staff']);
		$arr_role_id = new_explode(',', $aRow['role_id']);
		$arr_department_id = new_explode(',', $aRow['department_id']);
		$arr_staff_id = new_explode(',', $aRow['staff_id']);

		//check staff_department in department
		$check_staff_department = false;
		if(new_strlen($aRow['department_id']) > 0){
			if(count($staff_departments) > 0){
				foreach ($staff_departments as $staff_department) {
				    if(in_array($staff_department, $arr_department_id)){
				    	$check_staff_department = true;
				    }

				    if($check_staff_department){
				    	break;
				    }
				}
			}
		}

		if(in_array($staff_user_id, $arr_except_staff)){
			$output['iTotalRecords'] = (int)$output['iTotalRecords'] - 1;
			$output['iTotalDisplayRecords'] = (int)$output['iTotalDisplayRecords'] - 1;

			continue;
		}elseif( !(in_array($role_id, $arr_role_id) OR $check_staff_department OR in_array($staff_user_id, $arr_staff_id) OR ($aRow['is_public'] == 'yes') OR ($staff_user_id == $aRow['staff_create']) ) ){
			$output['iTotalRecords'] = (int)$output['iTotalRecords'] - 1;
			$output['iTotalDisplayRecords'] = (int)$output['iTotalDisplayRecords'] - 1;

			continue;
		}
		
	}

	for ($i = 0; $i < count($aColumns); $i++) {

		if($aColumns[$i] == db_prefix() . 'rb_templates.id') {
			$_data = $aRow['id'];

		}elseif ($aColumns[$i] == 'report_title') {
			$ask_user = rb_check_report_filter($aRow['id']);

			if($ask_user){

				$code = '<a href="javascript:void(0)" onclick="report_filter_modal('.$aRow['id'].'); return false;">' . $aRow['report_title'] . '</a>';
			}else{

				$code = '<a href="' . admin_url('report_builder/report_detail/' . $aRow['id']) . '">' . $aRow['report_title'] . '</a>';
			}
			$code .= '<div class="row-options">';

			$_data = $code;


		}elseif($aColumns[$i] == 'category_id'){
			$_data =  $aRow['name'];

		}elseif($aColumns[$i] == 'staff_create'){
			$_data =  get_staff_full_name($aRow['staff_create']);

		}elseif($aColumns[$i] == 'date_create'){

			$_data =  _dt($aRow['date_create']);

		}elseif($aColumns[$i] == '1'){
			$ask_user = rb_check_report_filter($aRow['id']);

			if($ask_user){

				$_data = '<a class="btn btn-primary btn-xs mleft5"  data-toggle="tooltip" title="" href="javascript:void(0)" onclick="report_filter_modal('.$aRow['id'].'); return false;" data-original-title="'._l('view').'"><i class="fa fa-eye"></i></a>';
			}else{

				$_data = '<a class="btn btn-primary btn-xs mleft5"  data-toggle="tooltip" title="" href="' . admin_url('report_builder/report_detail/' . $aRow['id']) . '"  data-original-title="'._l('view').'"><i class="fa fa-eye"></i></a>';
			}

			if($aRow['staff_create'] == get_staff_user_id()){

				$_data .= '<a class="btn btn-primary btn-xs mleft5"  data-toggle="tooltip" title="" href="javascript:void(0)" onclick="sharing_modal('.$aRow['id'].'); return false;" data-original-title="'._l('rb_sharing').'"><i class="fa fa-share"></i></a>';
			}

		}

		$row[] = $_data;
	}

	$output['aaData'][] = $row;
}

