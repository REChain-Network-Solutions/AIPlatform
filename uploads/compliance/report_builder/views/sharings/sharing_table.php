<?php

defined('BASEPATH') or exit('No direct script access allowed');

$aColumns = [
	db_prefix() . 'rb_templates.id',
	'report_title',
	'category_id',
	'role_id',
	'department_id',
	'staff_id',
	'except_staff',
	'1',
];
$sIndexColumn = 'id';
$sTable = db_prefix() . 'rb_templates';

$where = [];
$join = [
	'LEFT JOIN ' . db_prefix() . 'rb_categories ON ' . db_prefix() . 'rb_templates.category_id = ' . db_prefix() . 'rb_categories.id',
];

$category_filter = $this->ci->input->post('category_filter');
$role_filter = $this->ci->input->post('role_filter');
$department_filter = $this->ci->input->post('department_filter');
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

if(isset($role_filter)){
	$where_role = '';
	foreach ($role_filter as $role_id) {

		if($role_id != '')
		{
			if($where_role == ''){
				$where_role .= ' ('.'find_in_set('.$role_id.', '.db_prefix().'rb_templates.role_id)';
			}else{
				$where_role .= ' OR find_in_set('.$role_id.', '.db_prefix().'rb_templates.role_id)';
			}
		}
	}
	if($where_role != '')
	{
		$where_role .= ')';
		if($where != ''){
			array_push($where, 'AND'. $where_role);
		}else{
			array_push($where, $where_role);
		}
		
	}
}

//department_filter
if(isset($department_filter)){
	$where_department = '';
	foreach ($department_filter as $department_id) {

		if($department_id != '')
		{
			if($where_department == ''){
				$where_department .= ' ('.'find_in_set('.$department_id.', '.db_prefix().'rb_templates.department_id)';
			}else{
				$where_department .= ' OR find_in_set('.$department_id.', '.db_prefix().'rb_templates.department_id)';
			}
		}
	}
	if($where_department != '')
	{
		$where_department .= ')';
		if($where != ''){
			array_push($where, 'AND'. $where_department);
		}else{
			array_push($where, $where_department);
		}
		
	}
}

//staff_filter
if(isset($staff_filter)){
	$where_staff = '';
	foreach ($staff_filter as $staff_id) {

		if($staff_id != '')
		{
			if($where_staff == ''){
				$where_staff .= ' ('.'find_in_set('.$staff_id.', '.db_prefix().'rb_templates.staff_id)';
			}else{
				$where_staff .= ' OR find_in_set('.$staff_id.', '.db_prefix().'rb_templates.department_id)';
			}
		}
	}
	if($where_staff != '')
	{
		$where_staff .= ')';
		if($where != ''){
			array_push($where, 'AND'. $where_staff);
		}else{
			array_push($where, $where_staff);
		}
		
	}
}



if(!is_admin()){
	$where[] = 'AND '.db_prefix().'rb_templates.staff_create = '.get_staff_user_id();
}


$result = data_tables_init($aColumns, $sIndexColumn, $sTable, $join, $where, [db_prefix() . 'rb_templates.id', 'name']);

$output = $result['output'];
$rResult = $result['rResult'];

foreach ($rResult as $aRow) {
	$row = [];

	for ($i = 0; $i < count($aColumns); $i++) {

		if($aColumns[$i] == db_prefix() . 'rb_templates.id') {
			$_data = $aRow['id'];

		}elseif ($aColumns[$i] == 'report_title') {

			$code = '<a href="javascript:void(0)" onclick="sharing_modal('.$aRow['id'].'); return false;">' . $aRow['report_title'] . '</a>';
				
			$_data = $code;


		}elseif($aColumns[$i] == 'category_id'){
			$_data =  $aRow['name'];

		}elseif($aColumns[$i] == 'role_id'){

			$role_out_put = '';
			$role_ids       = new_strlen($aRow['role_id']) > 0 ? new_explode(',', $aRow['role_id']) : null;
			$list_role = '';

			if($role_ids != null){
				foreach ($role_ids as $key => $role_id) {
					$role_name   = rb_get_role_name($role_id);
					$list_role .= '<li class="text-success mbot10 mtop"><a href="#"  class="text-left">'.$role_name. '</a></li>';
				}
			}

			if($role_ids != null){
				$role_out_put .= '<span class="avatar bg-secondary brround avatar-none">+'. (count($role_ids) ) .'</span>';
			}

			$role_out_put1 = '<div class="task-info task-watched task-info-watched">
			<h5>
			<div class="btn-group">
			<span class="task-single-menu task-menu-watched">
			<div class="avatar-list avatar-list-stacked" data-toggle="dropdown">'.$role_out_put.'</div>
			<ul class="dropdown-menu list-staff" role="menu">
			<li class="dropdown-plus-title">
			'. _l('role') .'
			</li>
			'.$list_role.'
			</ul>
			</span>
			</div>
			</h5>
			</div>';

			$_data =  $role_out_put1;

		}elseif($aColumns[$i] == 'department_id'){

			$deparment_out_put = '';
			$department_ids       = new_strlen($aRow['department_id']) > 0 ? new_explode(',', $aRow['department_id']) : null;
			$list_deparment = '';
			
			if($department_ids != null){
				foreach ($department_ids as $key => $deparment_id) {
					$department_name   = rb_get_department_name($deparment_id);
					$list_deparment .= '<li class="text-success mbot10 mtop"><a href="#"  class="text-left">'.$department_name. '</a></li>';
				}
			}

			if($department_ids != null){
				$deparment_out_put .= '<span class="avatar bg-secondary brround avatar-none">+'. (count($department_ids) ) .'</span>';
			}

			$deparment_out_put1 = '<div class="task-info task-watched task-info-watched">
			<h5>
			<div class="btn-group">
			<span class="task-single-menu task-menu-watched">
			<div class="avatar-list avatar-list-stacked" data-toggle="dropdown">'.$deparment_out_put.'</div>
			<ul class="dropdown-menu list-staff" role="menu">
			<li class="dropdown-plus-title">
			'. _l('department') .'
			</li>
			'.$list_deparment.'
			</ul>
			</span>
			</div>
			</h5>
			</div>';

			$_data =  $deparment_out_put1;

		}elseif($aColumns[$i] == 'staff_id'){

			$staff_out_put = '';
			$staff_ids       = new_strlen($aRow['staff_id']) > 0 ? new_explode(',', $aRow['staff_id']) : null;
			$list_staff = '';
			
			if($staff_ids != null){
				foreach ($staff_ids as $key => $staff_id) {
					$staff_name   = get_staff_full_name($staff_id);
					$list_staff .= '<li class="text-success mbot10 mtop"><a href="#"  class="text-left">'.$staff_name. '</a></li>';
				}
			}

			if($staff_ids != null){
				$staff_out_put .= '<span class="avatar bg-secondary brround avatar-none">+'. (count($staff_ids) ) .'</span>';
			}

			$staff_out_put1 = '<div class="task-info task-watched task-info-watched">
			<h5>
			<div class="btn-group">
			<span class="task-single-menu task-menu-watched">
			<div class="avatar-list avatar-list-stacked" data-toggle="dropdown">'.$staff_out_put.'</div>
			<ul class="dropdown-menu list-staff" role="menu">
			<li class="dropdown-plus-title">
			'. _l('staff') .'
			</li>
			'.$list_staff.'
			</ul>
			</span>
			</div>
			</h5>
			</div>';
			$_data =  $staff_out_put1;

		}elseif($aColumns[$i] == 'except_staff'){

			$except_staff_out_put = '';
			$except_staff_ids       = new_strlen($aRow['except_staff']) > 0 ? new_explode(',', $aRow['except_staff']) : null;
			$list_except_staff = '';
			
			if($except_staff_ids != null){
				foreach ($except_staff_ids as $key => $staff_id) {
					$staff_name   = get_staff_full_name($staff_id);
					$list_except_staff .= '<li class="text-success mbot10 mtop"><a href="#"  class="text-left">'.$staff_name. '</a></li>';
				}
			}

			if($except_staff_ids != null){
				$except_staff_out_put .= '<span class="avatar bg-secondary brround avatar-none">+'. (count($except_staff_ids) ) .'</span>';
			}

			$except_staff_out_put1 = '<div class="task-info task-watched task-info-watched">
			<h5>
			<div class="btn-group">
			<span class="task-single-menu task-menu-watched">
			<div class="avatar-list avatar-list-stacked" data-toggle="dropdown">'.$except_staff_out_put.'</div>
			<ul class="dropdown-menu list-staff" role="menu">
			<li class="dropdown-plus-title">
			'. _l('rb_except_staff') .'
			</li>
			'.$list_except_staff.'
			</ul>
			</span>
			</div>
			</h5>
			</div>';
			$_data =  $except_staff_out_put1;

		}elseif($aColumns[$i] == '1'){

			$_data = '<a class="btn btn-primary btn-xs mleft5"  data-toggle="tooltip" title="" href="javascript:void(0)" onclick="sharing_modal('.$aRow['id'].'); return false;" data-original-title="'._l('rb_sharing').'"><i class="fa fa-share"></i></a>';

		}

		$row[] = $_data;
	}

	$output['aaData'][] = $row;
}

