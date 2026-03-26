<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>
<?php init_head(); ?>
<div id="wrapper">
	<div class="content">
		<div class="row">
			
			<div class="col-md-12" >
				<div class="panel_s">
					
					<div class="panel-body">
						<div class="row mb-5">

							<div class="col-md-3">
								<h4 class="no-margin"><i class="fa fa-list-alt" aria-hidden="true"></i>  <?php echo new_html_entity_decode($header); ?> </h4>
							</div>
							<!-- status -->
							<div class="col-md-9">
								<div class="sw-main sw-theme-arrows pull-right">

									<!-- SmartWizard html -->
									<ul class="nav nav-tabs step-anchor">
										<li class="active"><a href="<?php echo admin_url('report_builder/data_source_manage/'.$report_template_id.'?group=relationships'); ?>"><i class="fa fa-dashboard" aria-hidden="true"></i>  <?php echo _l('rb_data_source'); ?></a></li>
										<li class="step_active"><a href="#"><i class="fa fa-calculator" aria-hidden="true"></i>  <?php echo _l('rb_groups'); ?></a></li>
										<li class="step_active"><a href="#"><i class="fa fa-plus" aria-hidden="true"></i>  <?php echo _l('rb_subtotals'); ?></a></li>
										<li class="active"><a href="<?php echo admin_url('report_builder/column_manage/'.$report_template_id.'?group=column'); ?>"><i class="fa fa-th-large" aria-hidden="true"></i>  <?php echo _l('rb_columns'); ?></a></li>
										<li class="active"><a href="<?php echo admin_url('report_builder/cell_formatting/'.$report_template_id); ?>"><i class="fa fa-list-alt" aria-hidden="true"></i>  <?php echo _l('rb_cells'); ?></a></li>
										<li class=""><a href="#"><i class="fa fa-cog menu-icon menu-icon" aria-hidden="true"></i>  <?php echo _l('rb_settings'); ?></a></li>
									</ul>
								</div>

							</div>
							<!-- status -->
							
						</div>

						<!-- start tab -->
						<div class="modal-body">
							<div class="tab-content">
								<div class="row">
									<div>
										<div class="_buttons">
											<?php  if(is_admin() || has_permission('report_builder','','create')) { ?>
												<a href="#" onclick="add_cell_formatting(0, <?php echo new_html_entity_decode($report_template_id); ?>,' add'); return false;" class="btn btn-info mbot10"><?php echo _l('add_conditional_formatting'); ?></a>
											<?php } ?>
										</div>
										
										<div class="clearfix"></div>
										<br>
										<div>
											<?php 
											$table_data = array(
												_l('rb_condition_formatting'),
												_l('rb_color'),
												_l('option'),
											);

											render_datatable($table_data,'cell_format_table',
												array('customizable-table'),
												array(
													'id'=>'table-cell_format_table',
													'data-last-order-identifier'=>'cell_format_table',
													'data-default-order'=>get_table_last_order('cell_format_table'),
												)); 

												?>
										</div>
										<div id="modal_wrapper"></div>
									</div>
								</div>
							</div>
						</div>

							<div class="modal-footer">
								<a href="<?php echo admin_url('report_builder/report_manage'); ?>"  class="btn btn-default mr-2 "><?php echo _l('close'); ?></a>
								<a href="<?php echo admin_url('report_builder/column_manage/'.$report_template_id.'?group=column'); ?>"  class="btn btn-default mr-2 "><?php echo _l('rb_back'); ?></a>
								<button class="btn btn-info mr-2 data_source_next"><?php echo _l('rb_next'); ?></button>
							</div>

						</div>
					</div>
				</div>

			</div>
		</div>
		<?php echo form_hidden('report_template_id',$report_template_id); ?>


		<?php init_tail(); ?>
		<?php 

		require('modules/report_builder/assets/js/templates/cells/cell_format_manage_js.php');
	

		?>
	</body>

	</html>
