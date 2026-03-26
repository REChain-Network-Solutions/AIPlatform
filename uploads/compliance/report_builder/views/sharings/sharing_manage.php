<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php init_head(); ?>
<div id="wrapper">
	<div class="content">
		<div class="row">
			<div class="col-md-12">
				<div class="panel_s"> 
					<div class="panel-body">

						<div class="row">
							<div class="col-md-12">
								<h4 class="h4-color no-margin"><i class="fa fa-share-alt" aria-hidden="true"></i> <?php echo _l('rb_report_share'); ?></h4>
							</div>
						</div>
						<hr class="hr-color">

						<div class="row">
							
							<div  class="col-md-3 leads-filter-column">
								<select name="category_filter[]" id="category_filter" data-live-search="true" class="selectpicker" multiple="true" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('category'); ?>">
									<?php foreach($categories as $category) { ?>
										<option value="<?php echo new_html_entity_decode($category['id']); ?>"><?php echo new_html_entity_decode($category['name']); ?></option>
									<?php } ?>
								</select>
							</div>

							<div  class="col-md-3 leads-filter-column">
								<select name="role_filter[]" id="role_filter" data-live-search="true" class="selectpicker" multiple="true" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('role'); ?>">
									<?php foreach($roles as $role) { ?>
										<option value="<?php echo new_html_entity_decode($role['roleid']); ?>"><?php echo new_html_entity_decode($role['name']); ?></option>
									<?php } ?>
								</select>
							</div>
							 <div  class="col-md-3 leads-filter-column">
								<select name="department_filter[]" id="department_filter" data-live-search="true" class="selectpicker" multiple="true" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('department'); ?>">
									<?php foreach($departments as $department) { ?>
										<option value="<?php echo new_html_entity_decode($department['departmentid']); ?>"><?php echo new_html_entity_decode($department['name']); ?></option>
									<?php } ?>
								</select>
							</div>
							 <div  class="col-md-3 leads-filter-column">
								<select name="staff_filter[]" id="staff_filter" data-live-search="true" class="selectpicker" multiple="true" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('staff'); ?>">
									<?php foreach($staffs as $staff) { ?>
										<option value="<?php echo new_html_entity_decode($staff['staffid']); ?>"><?php echo new_html_entity_decode($staff['firstname'].' '.$staff['lastname']); ?></option>
									<?php } ?>
								</select>
							</div>
							 


						</div>
						<br>

						<?php render_datatable(array(
							_l('id'),
							_l('report_title'),
							_l('category_name'),
							_l('role'),
							_l('department'),
							_l('staff'),
							_l('rb_except_staff'),
							_l('rb_options'),
						),'sharing_table'); ?>
					</div>

					<div id="modal_wrapper"></div>
					

				</div>
			</div>

		</div>
	</div>
</div>
<?php init_tail(); ?>
<?php 
require('modules/report_builder/assets/js/sharings/sharing_manage_js.php');
?>
</body>
</html>

