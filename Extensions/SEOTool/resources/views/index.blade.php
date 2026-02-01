@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('SEO Tool'))
@section('titlebar_subtitle', __('Optimize content with our SEO Tool: keywords, meta titles, descriptions, and more. Integrated with Article Wizard for streamlined efficiency.'))

@section('content')
    <div
        class="py-10"
        x-data="{ 'activeFilter': 'Keywords' }"
    >
        <div class="container">
            @include('seo-tool::seo-tool')
        </div>
    </div>
@endsection

@push('script')
    <script src="/themes/default/assets/libs/apexcharts/dist/apexcharts.min.js"></script>
    <script>
        const filters = ['Keywords', 'Headers', 'Links', 'Images'];
        var total_percent = 0;
        var chart = undefined;
        var keywordsCount = 0;
        var headersCount = 0;
        var linksCount = 0;
        var imagesCount = 0;
        var keywords = [];
        var headers = [];
        var links = [];
        var images = [];

        function renderChart(percent, colorStops) {
            var options = {
                series: [percent],
                chart: {
                    type: 'radialBar',
                    offsetY: -20,
                    sparkline: {
                        enabled: true
                    }
                },
                plotOptions: {
                    radialBar: {
                        startAngle: -90,
                        endAngle: 90,
                        track: {
                            background: "hsl(var(--heading-foreground) / 3%)",
                            strokeWidth: '97%',
                            margin: 5,
                        },
                        dataLabels: {
                            enabled: false,
                        }
                    }
                },
                grid: {
                    padding: {
                        top: 10
                    }
                },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shade: 'light',
                        shadeIntensity: 0.4,
                        inverseColors: false,
                        opacityFrom: 1,
                        opacityTo: 1,
                        stops: [0, 30, 60, 100],
                        colorStops: [{
                                offset: 0,
                                color: colorStops[0],
                                opacity: 1
                            },
                            {
                                offset: 30,
                                color: colorStops[1],
                                opacity: 1
                            },
                            {
                                offset: 60,
                                color: colorStops[2] || colorStops[1],
                                opacity: 1
                            },
                            {
                                offset: 100,
                                color: colorStops[2] || colorStops[1],
                                opacity: 1
                            }
                        ]
                    },
                },
                labels: ['{{ __('SEO Percent') }}'],
                colors: ['hsl(var(--heading-foreground))'],
            };

            if (chart) {
                // remove the old chart
                chart.destroy();
                chart = new ApexCharts(document.getElementById('chart-credit'), options);
                chart.render();
            } else {
                chart = new ApexCharts(document.getElementById('chart-credit'), options);
                chart.render();
            }
            document.querySelector('.total_percent span').innerText = percent;
        }

        function updateWordCount() {
            let contentScan = $("#content_scan");
            let text = contentScan.text().trim();
            $('#content_length').text(text.length + ' / 20000');
        }
        document.addEventListener("DOMContentLoaded", function() {
            "use strict";
            var colorStops = ['#FF0000', '#FF0000'];
            renderChart(total_percent, colorStops);
            let content_scan = $("#content_scan");
            // update word
            let contentScan = $("#content_scan");
            contentScan.on('input', function() {
                updateWordCount();
            });
        });
    </script>

    <script>
        function sendScanRequest(ev) {
            "use strict";
            ev?.preventDefault();
            var spinner = document.querySelector('.refresh-icon');
            var improveSeoBtn = document.querySelector('#improve-seo-btn');
            var analyzSeoBtn = document.querySelector('#analys_btn');
            var reanalyzSeoBtn = document.querySelector('#reanalys_btn');
            var seoReportSection = document.querySelector('#seo_report');

            var resultText = $("#content_scan").html();
            let topic = $("#keyword");
            let type = $("#type");

            if (topic.val().length == 0) {
                toastr.warning('Please enter the keyword/topic.');
                return false;
            }
            if (resultText.length == 0) {
                toastr.warning('Please enter content.');
                return false;
            }
            if (resultText.length > 20000) {
                toastr.warning('The length of content should be 20000 characters or less.');
                return false;
            }

            imagesCount = getImagesCount(resultText);
            headersCount = getHeadersCount(resultText);
            linksCount = getLinksCount(resultText);

            var formData = new FormData();
            formData.append('topicKeyword', topic.val());
            formData.append('resultText', resultText);
            formData.append('type', type.val());
            formData.append('imagesCount', imagesCount);
            formData.append('headersCount', headersCount);
            formData.append('linksCount', linksCount);

            Alpine.store('appLoadingIndicator').show();
            $('#analys_btn').prop('disabled', true);
            spinner.classList.remove('hidden');
            $.ajax({
                type: "post",
                headers: {
                    'X-CSRF-TOKEN': "{{ csrf_token() }}",
                },
                url: "/dashboard/user/seo/analyseArticle",
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    seoReportSection.classList.remove('hidden');
                    updateSeoResult(data, imagesCount, headersCount, linksCount, resultText);
                    $('#analys_btn').prop('disabled', false);
                    spinner.classList.add('hidden');
                    Alpine.store('appLoadingIndicator').hide();
                    analyzSeoBtn.classList.add('hidden');
                    improveSeoBtn.classList.remove('hidden');
                    reanalyzSeoBtn.classList.remove('hidden');
                },
                error: function(data) {
                    console.log(data);
                    $('#analys_btn').prop('disabled', false);
                    spinner.classList.add('hidden');
                    Alpine.store('appLoadingIndicator').hide();
                }
            });
            return false;
        }

        function getImagesCount(content) {
            var img = content.match(/<img[^>]+>/g);
            var newimages = [];
            if (img) {
                for (var i = 0; i < img.length; i++) {
                    var src = img[i].match(/src="([^"]+)"/);
                    if (src) {
                        newimages.push(src[1]);
                    }
                }
            }
            return newimages.length || 0;
        }

        function getHeadersCount(content) {
            var newheaders = [];
            var headerPattern = /<h[1-6]>(.*?)<\/h[1-6]>/g;
            var match;
            while ((match = headerPattern.exec(content)) !== null) {
                newheaders.push(match[1]);
            }
            return newheaders.length || 0;
        }

        function getLinksCount(content) {
            var newlinks = [];
            var linkPattern = /<a.*?href="(.*?)".*?>(.*?)<\/a>/g;
            var match;
            while ((match = linkPattern.exec(content)) !== null) {
                newlinks.push(match[1]);
            }
            return newlinks.length || 0;
        }

        function updateSeoResult(data, imagesCount, headersCount, linksCount, resultText) {
            keywordsCount = 0;
            var competitorList = data.competitorList;
            var longTailList = data.longTailList;
            var containerCompetitorList = document.querySelector('.content_competitorList');
            var containerLongTailList = document.querySelector('.content_longTailList');

            containerCompetitorList.innerHTML = '';
            containerCompetitorList.innerHTML = '<div class="flex w-full flex-wrap gap-3" id="select_keywords">';
            for (let item of competitorList) {
                let matchData = checkKeywordMatch(item, resultText);
                if (matchData.isMatched) {
                    keywordsCount += matchData.matchCount;
                }
                let keyword =
                    `<button class="keyword me-1 my-1 ${matchData.matchCount > 0 ? 'bg-green-100 text-green-500' : 'bg-red-100 text-red-500'} cursor-pointer rounded-full border border-secondary px-3 py-1 font-medium">${item} <span class="text-xs font-normal text-heading">(${matchData.matchCount})</span></button>`;
                containerCompetitorList.innerHTML += keyword;
            }
            containerCompetitorList.innerHTML += '</div>';

            containerLongTailList.innerHTML = '';
            containerLongTailList.innerHTML = '<div class="flex w-full flex-wrap gap-3" id="select_keywords">';
            for (let item of longTailList) {
                let matchData = checkKeywordMatch(item, resultText);
                if (matchData.isMatched) {
                    keywordsCount += matchData.matchCount;
                }
                let keyword =
                    `<button class="keyword ${matchData.matchCount > 0 ? 'bg-green-100 text-green-500' : 'bg-red-100 text-red-500'} me-1 my-1 cursor-pointer rounded-full border border-secondary px-3 py-1 font-medium">${item} <span class="text-xs font-normal text-heading">(${matchData.matchCount})</span></button>`;
                containerLongTailList.innerHTML += keyword;
            }
            containerLongTailList.innerHTML += '</div>';



            filters.forEach(filter => {
                var count = document.querySelector('.count_' + filter);
                var numbers = count.querySelector('.numbers');
                // remove old results
                numbers.innerHTML = '';
                count.classList.remove('text-green-600', 'bg-green-500/10', 'text-red-700', 'bg-red-700/10', 'dark:bg-red-600/10', 'dark:text-red-600');
                numbers.innerHTML = window[filter.toLowerCase() + 'Count'];
                if (window[filter.toLowerCase() + 'Count'] > 0) {
                    count.classList.add('text-green-600', 'bg-green-500/10');
                    count.querySelector('.up').classList.remove('hidden');
                    if (!count.querySelector('.down').classList.contains('hidden')) {
                        count.querySelector('.down').classList.add('hidden');
                    }
                } else {
                    count.classList.add('text-red-700', 'bg-red-700/10', 'dark:bg-red-600/10', 'dark:text-red-600');
                    count.querySelector('.down').classList.remove('hidden');
                    if (!count.querySelector('.up').classList.contains('hidden')) {
                        count.querySelector('.up').classList.add('hidden');
                    }
                }
            });


            var per = data.percentage;
            if (per == null) {
                per = 0;
            }
            // convert per to int
            per = parseInt(per);

            var colorStops = [];
            if (per <= 30) {
                colorStops = ['#FF0000', '#FF0000']; // Red for 0-30%
            } else if (per <= 60) {
                colorStops = ['#FFA500', '#FFA500', '#FFA500']; // Red to Orange for 0-60%
            } else {
                colorStops = ['#1CA685', '#1CA685', '#1CA685']; // Red to Orange to Green for 0-90%
            }
            renderChart(per, colorStops);

        }

        function checkKeywordMatch(keyword, content) {
            // check match lowercase and uppercase
            var keywordLower = keyword.toLowerCase();
            var keywordUpper = keyword.toUpperCase();
            var keywordTitle = keyword.replace(/\b\w/g, l => l.toUpperCase());
            var keywordCapitalize = keyword.replace(/\b\w/g, l => l.toUpperCase());
            var keywordMatch = keyword + '|' + keywordLower + '|' + keywordUpper + '|' + keywordTitle + '|' + keywordCapitalize;
            var keywordPattern = new RegExp(keywordMatch, 'g');
            var matchCount = (content.match(keywordPattern) || []).length;
            var isMatched = matchCount > 0;
            return {
                isMatched: isMatched,
                matchCount: matchCount
            };
        }

        function improveSeo() {
            let controller = null; // Store the AbortController instance
            controller = new AbortController();
            const signal = controller.signal;
            let output = '';
            let chunk = [];

            let content_scan = $("#content_scan");
            let type = $("#type").val();

            var spinner = document.querySelector('.refresh-icon');
            var improveSeoBtn = document.querySelector('#improve-seo-btn');
            let topicKeyword = $("#keyword").val();

            spinner.classList.remove('hidden');
            Alpine.store('appLoadingIndicator').show();
            $('#improve-seo-btn').prop('disabled', true);

            let nIntervId = setInterval(function() {
                if (chunk.length == 0 && !streaming) {
                    clearInterval(nIntervId);
                    let percentage = document.querySelector('.total_percent span').innerText;
                    sendScanRequest();
                }
                const text = chunk.shift();
                if (text) {
                    output += text;
                    output = output.replace(/(<br>\s*){2,}/g, '<br>');
                    output = output.replace(/<h3>/g, '<br><br><h3>');
                    output = output.replace(/^(\s*<br\s*\/?>\s*)+(?=<h3>)/, '');
                    output = output.replace(/(<\/h3>\s*)(<br\s*\/?>\s*)+(?=\S)/g, '$1');
                    content_scan.html(output);
                }
            }, 20);

            var compatitorList = document.querySelectorAll('.content_competitorList button') || [];
            var longTailList = document.querySelectorAll('.content_longTailList button') || [];
            var imagesCount = getImagesCount(content_scan.html()) || 0;
            var headersCount = getHeadersCount(content_scan.html()) || 0;
            var linksCount = getLinksCount(content_scan.html()) || 0;

            var formData = new FormData();
            formData.append('topicKeyword', topicKeyword);
            formData.append('resultText', content_scan.html());
            formData.append('seoTool', true);
            formData.append('type', type);
            formData.append('competitorList', JSON.stringify(Array.from(compatitorList).map(item => item.innerText)));
            formData.append('longTailList', JSON.stringify(Array.from(longTailList).map(item => item.innerText)));
            formData.append('imagesCount', imagesCount);
            formData.append('headersCount', headersCount);
            formData.append('linksCount', linksCount);

            streaming = true;
            isGenerating = true;
            content_scan.html('');
            fetchEventSource('/dashboard/user/seo/improveArticle', {
                method: 'POST',
                headers: {
                    'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
                },
                body: formData,
                signal: signal,
                onmessage: (event) => {
                    if (event.data === '[DONE]') {
                        streaming = false;
                    }
                    if (event.data !== undefined && event.data !== null && event.data != '[DONE]') {
                        chunk.push(event.data.replace(/(?:\r\n|\r|\n)/g, ' <br> '));
                    }
                },
                onclose: () => {
                    streaming = false;
                    isGenerating = false;
                    spinner.classList.add('hidden');
                    $('#improve-seo-btn').prop('disabled', false);
                    Alpine.store('appLoadingIndicator').hide();
                },
                onerror: (err) => {
                    clearInterval(nIntervId);
                    streaming = false;
                    isGenerating = false;
                    console.log(err);
                    spinner.classList.add('hidden');
                    $('#improve-seo-btn').prop('disabled', false);
                    Alpine.store('appLoadingIndicator').hide();
                }
            });
        }
    </script>
@endpush
