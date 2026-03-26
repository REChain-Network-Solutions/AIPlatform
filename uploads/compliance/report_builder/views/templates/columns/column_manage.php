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
								<h4 class="no-margin"><i class="fa fa-th-large" aria-hidden="true"></i>  <?php echo new_html_entity_decode($header); ?> </h4>
							</div>
							<!-- status -->
							<div class="col-md-9">
								<div class="sw-main sw-theme-arrows pull-right">

									<!-- SmartWizard html -->
									<ul class="nav nav-tabs step-anchor">
										<li class="step_active"><a href="<?php echo admin_url('report_builder/data_source_manage/'.$report_template_id.'?group=relationships'); ?>"><i class="fa fa-dashboard" aria-hidden="true"></i>  <?php echo _l('rb_data_source'); ?></a></li>
										<li class="step_active"><a href="#"><i class="fa fa-calculator" aria-hidden="true"></i>  <?php echo _l('rb_groups'); ?></a></li>
										<li class="step_active"><a href="#"><i class="fa fa-plus" aria-hidden="true"></i>  <?php echo _l('rb_subtotals'); ?></a></li>
										<li class="step_active"><a href="<?php echo admin_url('report_builder/column_manage/'.$report_template_id.'?group=column'); ?>"><i class="fa fa-th-large" aria-hidden="true"></i>  <?php echo _l('rb_columns'); ?></a></li>
										<li class=""><a href="#"><i class="fa fa-list-alt" aria-hidden="true"></i>  <?php echo _l('rb_cells'); ?></a></li>
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
									<div class="horizontal-scrollable-tabs preview-tabs-top">

										<div class="scroller arrow-left"><i class="fa fa-angle-left"></i></div>
										<div class="scroller arrow-right"><i class="fa fa-angle-right"></i></div>
										<div class="horizontal-tabs">
											<ul class="nav nav-tabs nav-tabs-horizontal mbot15" role="tablist">
												<?php
												$i = 0;
												foreach($tab as $group_item){
													// if((count($group_by_columns) == 0 && $group_item == 'column') || $group_item != 'column'){
													?>
													<li <?php if($group_item == $group){echo " class='active'"; } ?>>
														<a href="<?php echo admin_url('report_builder/column_manage/'.$report_template_id.'?group='.$group_item); ?>" data-group="<?php echo new_html_entity_decode($group_item); ?>">

															<?php
															if($group_item == 'column'){
																echo '<i class="fa fa-th-large" aria-hidden="true"></i>  '._l('rb_columns');
															}elseif($group_item == 'label_cell_type'){
																echo '<i class="fa fa-pencil" aria-hidden="true"></i>  '._l('rb_label_cell_type');
															}elseif($group_item == 'aggregation_function'){
																echo '<i class="fa fa-bar-chart" aria-hidden="true"></i>  '._l('rb_aggregation_function');
															}
															?>
														</a>
													</li>
												<?php } ?>
											</ul>
										</div>
									</div>
										<div>
											<br>
											<?php $this->load->view($tabs['view']); ?>

										</div>
									</div>
								</div>
							</div>

							<div class="modal-footer">
								<a href="<?php echo admin_url('report_builder/report_manage'); ?>"  class="btn btn-default mr-2 "><?php echo _l('close'); ?></a>
								<a href="<?php echo admin_url('report_builder/subtotal_manage/'.$report_template_id); ?>"  class="btn btn-default mr-2 "><?php echo _l('rb_back'); ?></a>
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

		require('modules/report_builder/assets/js/templates/columns/column_manage_js.php');
		require('modules/report_builder/assets/js/templates/columns/column_js.php');
		require('modules/report_builder/assets/js/templates/columns/label_cell_type_js.php');

		?>
	</body>

	</html>
