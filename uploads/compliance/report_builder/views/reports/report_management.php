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
								<h4 class="h4-color no-margin"><i class="fa fa-list-alt menu-icon" aria-hidden="true"></i> <?php echo _l('report'); ?></h4>
							</div>
						</div>
						<hr class="hr-color">

						<?php if(has_permission('report_builder', '', 'create')){ ?>
							<div class="_buttons">
								<div class="btn-group">
									<a href="<?php echo admin_url('report_builder/add_data_source'); ?>"  class="btn btn-info pull-left display-block">
										<?php echo _l('rb_add_report'); ?>
									</a>
								</div>


								<div class="btn-group mleft5 hide">
										<a href="#" class="btn btn-info dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false"><?php echo _l('rb_add_report').' '; ?><span class="caret"></span></a>
										<ul class="dropdown-menu dropdown-menu-right">
											<li class="hidden-xs"><a href="<?php echo admin_url('report_builder/add_data_source'); ?>"  >
												<?php echo _l('rb_add_report_manually'); ?>
											</a>
										</li>
										
										<li class="hidden-xs"><a href="#" onclick="new_job_p(); return false;">
											<?php echo _l('rb_add_report_from_template'); ?></a>
										</li>
									</ul>
								</div>
							</div>
							<br>
						<?php } ?>

						<div class="row">
							
							<div  class="col-md-3 leads-filter-column">
								<select name="category_filter[]" id="category_filter" data-live-search="true" class="selectpicker" multiple="true" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('category'); ?>">
									<?php foreach($categories as $category) { ?>
										<option value="<?php echo new_html_entity_decode($category['id']); ?>"><?php echo new_html_entity_decode($category['name']); ?></option>
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
							_l('rb_staff_create'),
							_l('rp_date_create'),
							_l('rb_options'),

						),'report_table'); ?>
					</div>
					<div id="modal_wrapper"></div>
					

				</div>
			</div>

		</div>
	</div>
</div>
<?php init_tail(); ?>
<?php 
require('modules/report_builder/assets/js/reports/report_management_js.php');
?>
</body>
</html>
