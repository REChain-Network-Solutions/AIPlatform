<?php

declare(strict_types=1);

namespace App\Services\Security;

use Illuminate\Http\Client\Response;
use Illuminate\Support\Facades\Http;
use Illuminate\Validation\ValidationException;
use Psr\Http\Message\ResponseInterface;
use SimpleXMLElement;

class RssFeedService
{
    private const MAX_BYTES = 2 * 1024 * 1024;

    /** @return array<int, array{id: int, link: string, title: string, description: string, image: ?string}> */
    public function fetch(string $url, int $limit = 10): array
    {
        $response = $this->request($url);
        $body = $response->body();

        if (strlen($body) > self::MAX_BYTES) {
            throw ValidationException::withMessages(['url' => __('The RSS response is too large.')]);
        }

        $previous = libxml_use_internal_errors(true);

        try {
            $xml = simplexml_load_string($body, SimpleXMLElement::class, LIBXML_NONET|LIBXML_NOCDATA);
        } finally {
            libxml_use_internal_errors($previous);
            libxml_clear_errors();
        }

        if ($xml === false || ! isset($xml->channel->item)) {
            throw ValidationException::withMessages(['url' => __('The URL did not return a valid RSS feed.')]);
        }

        $posts = [];
        foreach ($xml->channel->item as $item) {
            if (count($posts) >= $limit) {
                break;
            }

            $posts[] = [
                'id'          => count($posts) + 1,
                'link'        => (string) $item->link,
                'title'       => (string) $item->title,
                'description' => (string) $item->description,
                'image'       => isset($item->enclosure['url']) ? (string) $item->enclosure['url'] : null,
            ];
        }

        return $posts;
    }

    private function request(string $url): Response
    {
        for ($redirects = 0; $redirects <= 3; $redirects++) {
            [$host, $port, $address] = $this->publicEndpoint($url);
            $resolvedAddress = str_contains($address, ':') ? '[' . $address . ']' : $address;
            $response = Http::accept('application/rss+xml, application/xml, text/xml')
                ->connectTimeout(3)
                ->timeout(8)
                ->withoutRedirecting()
                ->withOptions([
                    'curl'       => [CURLOPT_RESOLVE => [sprintf('%s:%d:%s', $host, $port, $resolvedAddress)]],
                    'on_headers' => static function (ResponseInterface $response): void {
                        $length = $response->getHeaderLine('Content-Length');
                        if ($length !== '' && (int) $length > self::MAX_BYTES) {
                            throw ValidationException::withMessages(['url' => __('The RSS response is too large.')]);
                        }
                    },
                    'progress' => static function (int $downloadTotal, int $downloadedBytes): void {
                        if ($downloadTotal > self::MAX_BYTES || $downloadedBytes > self::MAX_BYTES) {
                            throw ValidationException::withMessages(['url' => __('The RSS response is too large.')]);
                        }
                    },
                ])
                ->get($url);

            if (! $response->redirect()) {
                return $response->throw();
            }

            $location = $response->header('Location');
            if (! is_string($location) || ! preg_match('#^https?://#i', $location)) {
                throw ValidationException::withMessages(['url' => __('RSS redirects must use an absolute HTTP or HTTPS URL.')]);
            }

            $url = $location;
        }

        throw ValidationException::withMessages(['url' => __('The RSS URL redirected too many times.')]);
    }

    /** @return array{0: string, 1: int, 2: string} */
    private function publicEndpoint(string $url): array
    {
        $parts = parse_url($url);
        $scheme = strtolower((string) ($parts['scheme'] ?? ''));
        $host = strtolower(rtrim((string) ($parts['host'] ?? ''), '.'));

        if (! in_array($scheme, ['http', 'https'], true) || $host === '' || $host === 'localhost') {
            throw ValidationException::withMessages(['url' => __('Only public HTTP and HTTPS RSS URLs are allowed.')]);
        }

        $addresses = filter_var($host, FILTER_VALIDATE_IP) ? [$host] : $this->resolve($host);
        if ($addresses === []) {
            throw ValidationException::withMessages(['url' => __('The RSS hostname could not be resolved.')]);
        }

        foreach ($addresses as $address) {
            if (! filter_var($address, FILTER_VALIDATE_IP, FILTER_FLAG_NO_PRIV_RANGE|FILTER_FLAG_NO_RES_RANGE)) {
                throw ValidationException::withMessages(['url' => __('Private and reserved network addresses are not allowed.')]);
            }
        }

        $port = isset($parts['port']) ? (int) $parts['port'] : ($scheme === 'https' ? 443 : 80);

        return [$host, $port, $addresses[0]];
    }

    /** @return array<int, string> */
    private function resolve(string $host): array
    {
        $records = dns_get_record($host, DNS_A|DNS_AAAA);

        return array_values(array_unique(array_filter(array_map(
            static fn (array $record): ?string => $record['ip'] ?? $record['ipv6'] ?? null,
            $records ?: [],
        ))));
    }
}
