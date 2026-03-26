<div class="modal fade" id="sharingModal">
	<div class="modal-dialog ">
		<div class="modal-content">
			<div class="modal-header">
				<button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>

				<h4 class="modal-title"><?php echo new_html_entity_decode(_l('rb_sharing')); ?></h4>
			</div>
			<?php echo form_open(admin_url('report_builder/report_sharing_setting/'.$template_id), array('id' => 'rb_sharing')); ?>

			<?php 

			$layout_name = isset($get_report_setting) ? $get_report_setting->layout_name : 'align_left';
			$style_name = isset($get_report_setting) ? $get_report_setting->style_name : 'grey';
			$is_public = isset($get_report_setting) ? $get_report_setting->is_public : '';
			$role_id = isset($get_report_setting) ? new_explode(',', $get_report_setting->role_id) : '';
			$department_id = isset($get_report_setting) ? new_explode(',',$get_report_setting->department_id) : '';
			$staff_id = isset($get_report_setting) ? new_explode(',',$get_report_setting->staff_id) : '';
			$except_staff_id = isset($get_report_setting) ? new_explode(',',$get_report_setting->except_staff) : '';
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

			<?php if(isset($report)){ ?>
				<?php echo form_hidden('report',$report); ?>
			<?php } ?>

			<div class="modal-body">
				<div class="tab-content">

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
								<label for="specific_staff"><?php echo _l('specific_staff'); ?></label>
							</div>
						</div>
					</div>
				</div>

			<div class="col-md-12 option-show <?php echo new_html_entity_decode($option_show); ?>">
				<div class="row">
					<div class="col-md-12">
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
					<div class="col-md-12">
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
					<div class="col-md-12">
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

			<div class="row">
				<div class="col-md-12">
					<div class="form-group">
						<label for="except_staff" class="control-label"><?php echo _l('rb_except_staff'); ?></label>
						<select name="except_staff[]" class="form-control selectpicker" multiple="true" id="except_staff" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true"> 
							<?php foreach ($except_staffs as $key => $staff) { ?>
								<?php 
								$selected='';
								if(is_array($except_staff_id)){
									if(in_array($staff['staffid'], $except_staff_id)){
										$selected .= 'selected';
									}
								}
								?>

								<option value="<?php echo new_html_entity_decode($staff['staffid']); ?>" <?php echo new_html_entity_decode($selected); ?> ><?php  echo new_html_entity_decode($staff['firstname'].' '.$staff['lastname']); ?></option>
							<?php } ?>
						</select>
					</div>
				</div>
			</div>

					<div class="row">
						<div class="col-md-12">
							

						</div>
					</div>
				</div>
			</div>
			<div class="modal-footer">
				<button type="button" class="btn btn-default close_btn" data-dismiss="modal"><?php echo _l('close'); ?></button>
				<button type="submit" class="btn btn-info"><?php echo _l('submit'); ?></button>
			</div>

		</div>

		<?php echo form_close(); ?>
	</div>
</div>
</div>
<?php require('modules/report_builder/assets/js/sharings/sharing_modal_js.php'); ?>

