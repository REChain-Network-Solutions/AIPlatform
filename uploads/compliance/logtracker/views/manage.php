<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>
<?php init_head(); ?>

<?php
$dateColor = getColorByCategory('date');
$totalColor = getColorByCategory('all');
$errorColor = getColorByCategory('error');
$debugColor = getColorByCategory('debug');
$infoColor = getColorByCategory('info');
?>


<div id="wrapper">
	<div class="content">
		<div class="row">
			<div class="col-md-12">
				<div class="panel_s">
					<div class="panel-body">
						<h4 class="tw-mt-0 tw-font-semibold tw-text-lg tw-flex tw-items-center">
							<span><i class="fa-solid fa-house"></i>
								<?php echo _l('logtracker_dashboard') ?>
							</span>
						</h4>
						<hr class="hr-panel-heading">
						<table class="table dt-table" data-order-type="asc" data-order-col="2" id="consentHistoryTable">
							<thead>
								<tr>
									<th>
										<?= _l('date') ?>
									</th>
									<th>
										<?= _l('all') ?>
									</th>
									<th>
										<?= _l('error') ?>
									</th>
									<th>
										<?= _l('debug') ?>
									</th>
									<th>
										<?= _l('info') ?>
									</th>
									<th>
										<?= _l('actions') ?>
									</th>
								</tr>
							</thead>
							<tbody>
								<?php foreach ($summary['data'] as $key => $value) {
									$date = '<span class="label" style="color:' . $dateColor . ';border:1px solid ' . adjust_hex_brightness($dateColor, 0.4) . ';background: ' . adjust_hex_brightness($dateColor, 0.04) . ';">';
									$date .= _d($key);
									$date .= '</span>';

									$total = '<span class="label" style="color:' . $totalColor . ';border:1px solid ' . adjust_hex_brightness($totalColor, 0.4) . ';background: ' . adjust_hex_brightness($totalColor, 0.04) . ';">';
									$total .= $value['count'];
									$total .= '</span>';

									$error = '<span class="label" style="color:' . $errorColor . ';border:1px solid ' . adjust_hex_brightness($errorColor, 0.4) . ';background: ' . adjust_hex_brightness($errorColor, 0.04) . ';">';
									$error .= $value['data']["ERROR"]['count'] ?? 0;
									$error .= '</span>';

									$debug = '<span class="label" style="color:' . $debugColor . ';border:1px solid ' . adjust_hex_brightness($debugColor, 0.4) . ';background: ' . adjust_hex_brightness($debugColor, 0.04) . ';">';
									$debug .= $value['data']["DEBUG"]['count'] ?? 0;
									$debug .= '</span>';

									$info = '<span class="label" style="color:' . $infoColor . ';border:1px solid ' . adjust_hex_brightness($infoColor, 0.4) . ';background: ' . adjust_hex_brightness($infoColor, 0.04) . ';">';
									$info .= $value['data']["INFO"]['count'] ?? 0;
									$info .= '</span>';

									$actions = '';
									$actions .= '<div class="tw-flex tw-items-center tw-space-x-3">';
									$actions .= '<a href="' . admin_url('logtracker/view/') . $key . '" class="text-info" data-toggle="tooltip" data-title="' . _l('view_log_details') . '">
										<i class="fa-solid fa-eye fa-lg"></i>
									</a>';
									if (has_permission('logtracker', '', 'download')) {
										$actions .= '<a href="' . admin_url('logtracker/downloadLogFile/') . 'log-' . $key . '" class="text-success" data-toggle="tooltip" data-title="' . _l('download_log_file') . '">
											<i class="fa-solid fa-file-arrow-down fa-lg"></i>
										</a>';
										$actions .= '<a href="' . admin_url('logtracker/downloadLogFile/') . 'log-' . $key . '/true' . '" class="text-warning" data-toggle="tooltip" data-title="' . _l('download_log_file_as_a_zip') . '">
											<i class="fa-solid fa-file-zipper fa-lg"></i>
										</a>';
									}
									if (has_permission('logtracker', '', 'delete')) {
										$actions .= '<a href="javascript:void(0)" class="text-danger" onclick="deleteLogFile(\'' . 'log-' . $key . '\')" data-toggle="tooltip" data-title="' . _l('delete_log_file') . '">
											<i class="fa-regular fa-trash-can fa-lg"></i>
										</a>';
									}
									$actions .= '</div>';
									?>
									<tr>
										<td>
											<?= $date ?>
										</td>
										<td>
											<?= $total ?>
										</td>
										<td>
											<?= $error ?>
										</td>
										<td>
											<?= $debug ?>
										</td>
										<td>
											<?= $info ?>
										</td>
										<td>
											<?= $actions ?>
										</td>
									</tr>
								<?php } ?>
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>
<?php init_tail(); ?>

<script>
	"use strict";

	$(function () {
		initDataTableInline('.table-logtracker');
	});
</script>