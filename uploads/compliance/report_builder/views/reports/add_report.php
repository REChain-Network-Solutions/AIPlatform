<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php init_head(); ?>
<div id="wrapper">
	<div class="content">
		<div class="row">
			<?php 
			$id = '';
			$title = '';
			$title .= _l('view_manufacturing_order_lable');

			$start_working_hide='';
			$action_hide='';
			$cancel_hide='';

			$waiting_for_another_wo_active='';
			$ready_active='';
			$progress_active='';
			$finished_active='';


			?>

			<div class="col-md-12" >
				<div class="panel_s">
					
					<div class="panel-body">
						<div class="row mb-5">
							<div class="col-md-6">
								<h4 class="no-margin"><?php echo new_html_entity_decode($header); ?> 
							</div>
							<!-- status -->
							<div class="col-md-6">
								<div class="sw-main sw-theme-arrows pull-right">

									<!-- SmartWizard html -->
									<ul class="nav nav-tabs step-anchor">
										<li class="<?php echo new_html_entity_decode($waiting_for_another_wo_active) ?>"><a href="#"><?php echo _l('rb_data_source'); ?></a></li>
										<li class="<?php echo new_html_entity_decode($ready_active) ?>"><a href="#"><?php echo _l('rb_columns'); ?></a></li>
										<li class="<?php echo new_html_entity_decode($progress_active) ?>"><a href="#"><?php echo _l('rb_cells'); ?></a></li>
										<li class="<?php echo new_html_entity_decode($finished_active) ?>"><a href="#"><?php echo _l('rb_groups'); ?></a></li>
										<li class="<?php echo new_html_entity_decode($finished_active) ?>"><a href="#"><?php echo _l('rb_settings'); ?></a></li>
									</ul>
								</div>

							</div>
							<!-- status -->
							
						</div>
						<br>
						<hr class="hr-color no-margin">
						<br>

						<!-- action related work order -->
						<div class="row">
							
							

						</div>
						<!-- action related work order -->


						<!-- start tab -->
						<div class="modal-body">
							<div class="tab-content">
								<!-- start general infor -->
								<?php 

								?>
								<div class="row">
									<div class="col-md-6 panel-padding" >
										<input type="hidden" name="manufacturing_order" value="<?php echo new_html_entity_decode('11') ?>">
										<input type="hidden" name="work_order_id" value="<?php echo new_html_entity_decode('11') ?>">

										<table class="table border table-striped table-margintop" >
											<tbody>
												<tr class="project-overview">
													<td class="bold td-width"><?php echo _l('to_produce'); ?></td>
													<td><b><?php echo 'to_produce' ; ?></b></td>
												</tr>
												<tr class="project-overview">
													<td class="bold"><?php echo _l('quantity_produced'); ?></td>
													<td><?php echo new_html_entity_decode('11') ; ?><b></b></td>
												</tr>

											</tbody>
										</table>
									</div>

								</div>

							</div>

							<div class="modal-footer">
								<a href="<?php echo admin_url('manufacturing/work_order_manage'); ?>"  class="btn btn-default mr-2 "><?php echo _l('close'); ?></a>
								<a href="<?php echo admin_url('manufacturing/work_order_manage'); ?>"  class="btn btn-default mr-2 "><?php echo _l('rb_back'); ?></a>
								<a href="<?php echo admin_url('manufacturing/work_order_manage'); ?>"  class="btn btn-info mr-2 "><?php echo _l('rb_next'); ?></a>

							</div>

						</div>
					</div>
				</div>

			</div>
		</div>
		<?php init_tail(); ?>
		
	</body>
	</html>
