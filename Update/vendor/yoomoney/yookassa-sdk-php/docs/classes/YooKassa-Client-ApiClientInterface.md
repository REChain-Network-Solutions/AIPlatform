# [YooKassa API SDK](../home.md)

# Interface: ApiClientInterface
### Namespace: [\YooKassa\Client](../namespaces/yookassa-client.md)
---
**Summary:**

Interface ApiClientInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [call()](../classes/YooKassa-Client-ApiClientInterface.md#method_call) |  | Создает CURL запрос, получает и возвращает обработанный ответ |
| public | [getUserAgent()](../classes/YooKassa-Client-ApiClientInterface.md#method_getUserAgent) |  | Возвращает UserAgent. |
| public | [setAdvancedCurlOptions()](../classes/YooKassa-Client-ApiClientInterface.md#method_setAdvancedCurlOptions) |  | Устанавливает дополнительные настройки curl. |
| public | [setBearerToken()](../classes/YooKassa-Client-ApiClientInterface.md#method_setBearerToken) |  | Устанавливает OAuth-токен магазина. |
| public | [setConfig()](../classes/YooKassa-Client-ApiClientInterface.md#method_setConfig) |  | Устанавливает настройки. |
| public | [setLogger()](../classes/YooKassa-Client-ApiClientInterface.md#method_setLogger) |  | Устанавливает объект для логирования. |
| public | [setShopId()](../classes/YooKassa-Client-ApiClientInterface.md#method_setShopId) |  | Устанавливает shopId магазина. |
| public | [setShopPassword()](../classes/YooKassa-Client-ApiClientInterface.md#method_setShopPassword) |  | Устанавливает секретный ключ магазина. |

---
### Details
* File: [lib/Client/ApiClientInterface.php](../../lib/Client/ApiClientInterface.php)
* Package: \YooKassa
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |

---
## Methods
<a name="method_call" class="anchor"></a>
#### public call() : \YooKassa\Common\ResponseObject

```php
public call(string $path, string $method, array $queryParams, null|string $httpBody = null, array $headers = []) : \YooKassa\Common\ResponseObject
```

**Summary**

Создает CURL запрос, получает и возвращает обработанный ответ

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | path  | URL запроса |
| <code lang="php">string</code> | method  | HTTP метод |
| <code lang="php">array</code> | queryParams  | Массив GET параметров запроса |
| <code lang="php">null OR string</code> | httpBody  | Тело запроса |
| <code lang="php">array</code> | headers  | Массив заголовков запроса |

**Returns:** \YooKassa\Common\ResponseObject - 


<a name="method_setLogger" class="anchor"></a>
#### public setLogger() : mixed

```php
public setLogger(null|\Psr\Log\LoggerInterface $logger) : mixed
```

**Summary**

Устанавливает объект для логирования.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \Psr\Log\LoggerInterface</code> | logger  | Объект для логирования |

**Returns:** mixed - 


<a name="method_getUserAgent" class="anchor"></a>
#### public getUserAgent() : \YooKassa\Client\UserAgent

```php
public getUserAgent() : \YooKassa\Client\UserAgent
```

**Summary**

Возвращает UserAgent.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

**Returns:** \YooKassa\Client\UserAgent - 


<a name="method_setShopId" class="anchor"></a>
#### public setShopId() : mixed

```php
public setShopId(int|string|null $shopId) : mixed
```

**Summary**

Устанавливает shopId магазина.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR string OR null</code> | shopId  | shopId магазина |

**Returns:** mixed - 


<a name="method_setShopPassword" class="anchor"></a>
#### public setShopPassword() : mixed

```php
public setShopPassword(?string $shopPassword) : mixed
```

**Summary**

Устанавливает секретный ключ магазина.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?string</code> | shopPassword  |  |

**Returns:** mixed - 


<a name="method_setBearerToken" class="anchor"></a>
#### public setBearerToken() : mixed

```php
public setBearerToken(?string $bearerToken) : mixed
```

**Summary**

Устанавливает OAuth-токен магазина.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?string</code> | bearerToken  |  |

**Returns:** mixed - 


<a name="method_setConfig" class="anchor"></a>
#### public setConfig() : mixed

```php
public setConfig(array $config) : mixed
```

**Summary**

Устанавливает настройки.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | config  |  |

**Returns:** mixed - 


<a name="method_setAdvancedCurlOptions" class="anchor"></a>
#### public setAdvancedCurlOptions() : void

```php
public setAdvancedCurlOptions() : void
```

**Summary**

Устанавливает дополнительные настройки curl.

**Details:**
* Inherited From: [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

**Returns:** void - 




---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney