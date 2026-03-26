<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php if($general_setting_href == 'general_setting'){ ?>
<div class="row">
	<div class="col-md-12">
		<h4 class="h4-color"><i class="fa fa-bars menu-icon" aria-hidden="true"></i> <?php echo _l('general_setting'); ?></h4>
	</div>
</div>
<?php } ?>

<?php 
	$template_id='';
	if(isset($report_template_id)){
		$template_id = $report_template_id;
	}
 ?>

<?php echo form_open_multipart(admin_url('report_builder/'.$general_setting_href.'/'.$template_id),array('class'=>'general_setting','autocomplete'=>'off')); ?>
<div class="row">
	<div class="col-md-12">
		<div class="horizontal-scrollable-tabs preview-tabs-top">
			<div class="scroller arrow-left"><i class="fa fa-angle-left"></i></div>
			<div class="scroller arrow-right"><i class="fa fa-angle-right"></i></div>
			<div class="horizontal-tabs">
				<ul class="nav nav-tabs nav-tabs-horizontal mbot15" role="tablist">
					<li role="presentation" class="active">
						<a href="#rb_appearance" aria-controls="rb_appearance"  class="rb_appearance" role="tab" data-toggle="tab">
							<span class="glyphicon glyphicon-align-justify"></span>&nbsp;<?php echo _l('rb_appearance'); ?>
						</a>
					</li>
					<li role="presentation" class="">
						<a href="#rb_security" aria-controls="rb_security" role="tab" data-toggle="tab">
							<span class="fa fa-cogs menu-icon"></span>&nbsp;<?php echo _l('rb_security'); ?>
						</a>
					</li>
					<li role="presentation" >
						<a href="#rb_titles" aria-controls="rb_titles" role="tab" data-toggle="tab">
							<span class="fa fa-balance-scale menu-icon"></span>&nbsp;<?php echo _l('rb_titles'); ?>
						</a>
					</li>
				</ul>
			</div>
		</div>
		<br>

		<?php 

			$layout_name = isset($get_report_setting) ? $get_report_setting->layout_name : 'align_left';
			$style_name = isset($get_report_setting) ? $get_report_setting->style_name : 'grey';
			$is_public = isset($get_report_setting) ? $get_report_setting->is_public : '';
			$role_id = isset($get_report_setting) ? new_explode(',', $get_report_setting->role_id) : '';
			$department_id = isset($get_report_setting) ? new_explode(',',$get_report_setting->department_id) : '';
			$staff_id = isset($get_report_setting) ? new_explode(',',$get_report_setting->staff_id) : '';
			$report_title = isset($get_report_setting) ? $get_report_setting->report_title : $report_title_sample;
			$category_id = isset($get_report_setting) ? $get_report_setting->category_id : '';
			$report_footer = isset($get_report_setting) ? $get_report_setting->report_footer : '';
			$report_header = isset($get_report_setting) ? $get_report_setting->report_header : '';
			$report_name = isset($get_report_setting) ? $get_report_setting->report_name : $report_title_sample;
			$records_per_page = isset($get_report_setting) ? $get_report_setting->records_per_page : 20;

			if(isset($get_report_setting)){
				if($get_report_setting->is_public == 'yes'){
					$option_show = ' hide';

				}else{
					$option_show = '';
				}
			}else{
				$option_show = ' hide';
			}

		 ?>

		<div class="col-md-12">
			<div class="tab-content ">
				<div role="tabpanel" class="tab-pane active" id="rb_appearance">
					<h5 class="font-bold "><?php echo _l('lay_out') ?></h5>

					<div class="row">
						<div class="col-md-12">
							<div class="form-group">
								<div class="radio radio-primary radio-inline" >
									<input type="radio" id="align_left" name="layout_name" value="align_left" <?php if($layout_name == 'align_left'){ echo "checked" ;}; ?>>
									<label for="align_left"><?php echo _l('align_left'); ?></label>
								</div>
							</div>
						</div>
					</div>
					<div class="row">
						<div class="col-md-12">
							<div class="form-group">
								<div class="radio radio-primary radio-inline" >
									<input type="radio" id="align_right" name="layout_name" value="align_right" <?php if($layout_name == 'align_right'){ echo "checked" ;}; ?>>
									<label for="align_right"><?php echo _l('align_right'); ?></label>
								</div>
							</div>
						</div>
					</div>

					<h5 class="font-bold "><?php echo _l('rb_styles') ?></h5>

					<div class="row">
						<div class="col-md-5">
							<div class="form-group">
								<select name="style_name" id="style_name" class="selectpicker"  data-width="100%" data-none-selected-text="<?php echo _l('Alert'); ?>">
									<option value="blue" <?php if($style_name == 'blue'){ echo "selected" ;}; ?> ><?php echo _l('blue') ; ?></option>
									<option value="grey" <?php if($style_name == 'grey'){ echo "selected" ;}; ?> ><?php echo _l('grey') ; ?></option>
									<option value="teal" <?php if($style_name == 'teal'){ echo "selected" ;}; ?> ><?php echo _l('teal') ; ?></option>
								</select>
							</div>
						</div>
					</div>

				</div>
				<div role="tabpanel" class="tab-pane " id="rb_security">
					<h5 class="font-bold "><?php echo _l('who_can_access_the_generated_report') ?></h5>
					<h5 class="font-bold font-style-italic"><?php echo _l('the_admin_and_user_created_this_report_can_access_this_report') ?></h5>

					<div class="row">
						<div class="col-md-12">
							<div class="form-group">
								<div class="radio radio-primary radio-inline">
									<input  type="radio" id="is_public" name="is_public" value="yes" <?php if($is_public == 'yes'){ echo "checked" ;}; ?> >
									<label for="is_public"><?php echo _l('this_report_is_public '); ?>
									<a href="#" class="pull-right display-block input_method"> <i class="fa fa-question-circle i_tooltip" data-toggle="tooltip" title="" data-original-title="<?php echo _l('anyone_on_the_Internet_can_access_this_report'); ?>"></i></a>
								</label>
							</div>
						</div>
					</div>
				</div>


				<div class="row">
					<div class="col-md-12">
						<div class="form-group">
							<div class="radio radio-primary radio-inline">
								<input type="radio" id="specific_staff" name="is_public"  value="no" <?php if($is_public == 'no'){ echo "checked" ;}; ?>>
								<label for="specific_staff"><?php echo _l('specific_staff'); ?>
							</label>
						</div>
					</div>
				</div>
			</div>

			<div class="col-md-12 option-show <?php echo new_html_entity_decode($option_show); ?>">
				<div class="row">
					<div class="col-md-6">
						<div class="form-group">
							<label for="role_id" class="control-label"><?php echo _l('role'); ?></label>
							<select name="role_id[]" class="form-control selectpicker" multiple="true" id="role_id" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true"> 
								<?php foreach ($roles as $key => $role) { ?>
									<?php 
									$selected='';
										if(is_array($role_id)){
											if(in_array($role['roleid'], $role_id)){
												$selected .= 'selected';
											}
										}
									 ?>
									 
									<option value="<?php echo new_html_entity_decode($role['roleid']); ?>" <?php echo new_html_entity_decode($selected); ?>><?php  echo new_html_entity_decode($role['name']); ?></option>
								<?php } ?>
							</select>

						</div>
					</div>
				</div>
				<div class="row">
					<div class="col-md-6">
						<div class="form-group">
							<label for="department_id" class="control-label"><?php echo _l('department_name'); ?></label>
							<select name="department_id[]" class="form-control selectpicker" multiple="true" id="department_id" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true"> 
								<?php foreach ($departments as $department_key => $department) { ?>

									<?php 
									$selected='';
										if(is_array($department_id)){
											if(in_array($department['departmentid'], $department_id)){
												$selected .= 'selected';
											}
										}
									 ?>

									<option value="<?php echo new_html_entity_decode($department['departmentid']); ?>" <?php echo new_html_entity_decode($selected); ?>><?php  echo new_html_entity_decode($department['name']); ?></option>
								<?php } ?>
							</select>
						</div>
					</div>
				</div>


				<div class="row">
					<div class="col-md-6">
						<div class="form-group">
							<label for="staff_id" class="control-label"><?php echo _l('staff'); ?></label>
							<select name="staff_id[]" class="form-control selectpicker" multiple="true" id="staff_id" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true"> 
								<?php foreach ($staffs as $key => $staff) { ?>

									<?php 
									$selected='';
										if(is_array($staff_id)){
											if(in_array($staff['staffid'], $staff_id)){
												$selected .= 'selected';
											}
										}
									 ?>

									<option value="<?php echo new_html_entity_decode($staff['staffid']); ?>" <?php echo new_html_entity_decode($selected); ?>><?php  echo new_html_entity_decode($staff['firstname'].' '.$staff['lastname']); ?></option>
								<?php } ?>
							</select>
						</div>
					</div>
				</div>
			</div>

		</div>
		<div role="tabpanel" class="tab-pane " id="rb_titles">
			<div class="row">
				<div class="col-md-6">
					<?php echo render_input('report_title', 'report_title', $report_title); ?>
				</div>
				<div class="col-md-6">
					<?php echo render_select('category_id',$categories,array('id','name'),'category_name', $category_id); ?>
				</div>
				<div class="col-md-12"> 
					<?php echo render_textarea('report_footer','report_footer', $report_footer); ?>
				</div>
				<div class="col-md-12"> 
					<?php echo render_textarea('report_header','report_header', $report_header); ?>
				</div>
				<div class="col-md-12">
					<?php echo render_input('report_name', 'report_name', $report_name); ?>
				</div>
				<div class="col-md-12">
					<?php echo render_input('records_per_page', 'records_per_page',$records_per_page, 'number'); ?>
				</div>

			</div>
		</div>
	</div>
</div>

</div>
</div>


<div class="clearfix"></div>

<div class="modal-footer">
	<?php if(has_permission('report_builder', '', 'create') || has_permission('report_builder', '', 'edit') ){ ?>
		<button type="submit" class="btn btn-info"><?php if(isset($get_report_setting)){echo _l('update');}else{ echo _l('submit');} ?></button>
	<?php } ?>
</div>
<?php echo form_close(); ?>


</body>
</html>


