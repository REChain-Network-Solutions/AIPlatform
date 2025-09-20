# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Client\CurlClient
### Namespace: [\YooKassa\Client](../namespaces/yookassa-client.md)
---
**Summary:**

Класс, представляющий модель CurlClient.

**Description:**

Класс клиента Curl запросов.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Client-CurlClient.md#method___construct) |  | CurlClient constructor. |
| public | [call()](../classes/YooKassa-Client-CurlClient.md#method_call) |  | Создает CURL запрос, получает и возвращает обработанный ответ |
| public | [closeCurlConnection()](../classes/YooKassa-Client-CurlClient.md#method_closeCurlConnection) |  | Close connection. |
| public | [getConfig()](../classes/YooKassa-Client-CurlClient.md#method_getConfig) |  | Возвращает настройки. |
| public | [getConnectionTimeout()](../classes/YooKassa-Client-CurlClient.md#method_getConnectionTimeout) |  | Возвращает значение параметра CURLOPT_CONNECTTIMEOUT. |
| public | [getProxy()](../classes/YooKassa-Client-CurlClient.md#method_getProxy) |  | Возвращает настройки прокси. |
| public | [getTimeout()](../classes/YooKassa-Client-CurlClient.md#method_getTimeout) |  | Возвращает значение параметра CURLOPT_TIMEOUT. |
| public | [getUserAgent()](../classes/YooKassa-Client-CurlClient.md#method_getUserAgent) |  | Возвращает UserAgent. |
| public | [sendRequest()](../classes/YooKassa-Client-CurlClient.md#method_sendRequest) |  | Выполняет запрос, получает и возвращает обработанный ответ |
| public | [setAdvancedCurlOptions()](../classes/YooKassa-Client-CurlClient.md#method_setAdvancedCurlOptions) |  | Устанавливает дополнительные настройки curl. |
| public | [setBearerToken()](../classes/YooKassa-Client-CurlClient.md#method_setBearerToken) |  | Устанавливает OAuth-токен магазина. |
| public | [setBody()](../classes/YooKassa-Client-CurlClient.md#method_setBody) |  | Устанавливает тело запроса. |
| public | [setConfig()](../classes/YooKassa-Client-CurlClient.md#method_setConfig) |  | Устанавливает настройки. |
| public | [setConnectionTimeout()](../classes/YooKassa-Client-CurlClient.md#method_setConnectionTimeout) |  | Устанавливает значение параметра CURLOPT_CONNECTTIMEOUT. |
| public | [setCurlOption()](../classes/YooKassa-Client-CurlClient.md#method_setCurlOption) |  | Устанавливает параметры CURL. |
| public | [setKeepAlive()](../classes/YooKassa-Client-CurlClient.md#method_setKeepAlive) |  | Устанавливает флаг сохранения соединения. |
| public | [setLogger()](../classes/YooKassa-Client-CurlClient.md#method_setLogger) |  | Устанавливает объект для логирования. |
| public | [setProxy()](../classes/YooKassa-Client-CurlClient.md#method_setProxy) |  | Устанавливает настройки прокси. |
| public | [setShopId()](../classes/YooKassa-Client-CurlClient.md#method_setShopId) |  | Устанавливает shopId магазина. |
| public | [setShopPassword()](../classes/YooKassa-Client-CurlClient.md#method_setShopPassword) |  | Устанавливает секретный ключ магазина. |
| public | [setTimeout()](../classes/YooKassa-Client-CurlClient.md#method_setTimeout) |  | Устанавливает значение параметра CURLOPT_TIMEOUT. |

---
### Details
* File: [lib/Client/CurlClient.php](../../lib/Client/CurlClient.php)
* Package: YooKassa
* Class Hierarchy:
  * \YooKassa\Client\CurlClient
* Implements:
  * [\YooKassa\Client\ApiClientInterface](../classes/YooKassa-Client-ApiClientInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct() : mixed
```

**Summary**

CurlClient constructor.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** mixed - 


<a name="method_call" class="anchor"></a>
#### public call() : \YooKassa\Common\ResponseObject

```php
public call(string $path, string $method, array $queryParams, null|string $httpBody = null, array $headers = []) : \YooKassa\Common\ResponseObject
```

**Summary**

Создает CURL запрос, получает и возвращает обработанный ответ

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | path  | URL запроса |
| <code lang="php">string</code> | method  | HTTP метод |
| <code lang="php">array</code> | queryParams  | Массив GET параметров запроса |
| <code lang="php">null OR string</code> | httpBody  | Тело запроса |
| <code lang="php">array</code> | headers  | Массив заголовков запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ApiException |  |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException |  |

**Returns:** \YooKassa\Common\ResponseObject - 


<a name="method_closeCurlConnection" class="anchor"></a>
#### public closeCurlConnection() : void

```php
public closeCurlConnection() : void
```

**Summary**

Close connection.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** void - 


<a name="method_getConfig" class="anchor"></a>
#### public getConfig() : array

```php
public getConfig() : array
```

**Summary**

Возвращает настройки.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** array - 


<a name="method_getConnectionTimeout" class="anchor"></a>
#### public getConnectionTimeout() : int

```php
public getConnectionTimeout() : int
```

**Summary**

Возвращает значение параметра CURLOPT_CONNECTTIMEOUT.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** int - 


<a name="method_getProxy" class="anchor"></a>
#### public getProxy() : string

```php
public getProxy() : string
```

**Summary**

Возвращает настройки прокси.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** string - 


<a name="method_getTimeout" class="anchor"></a>
#### public getTimeout() : int

```php
public getTimeout() : int
```

**Summary**

Возвращает значение параметра CURLOPT_TIMEOUT.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** int - 


<a name="method_getUserAgent" class="anchor"></a>
#### public getUserAgent() : \YooKassa\Client\UserAgent

```php
public getUserAgent() : \YooKassa\Client\UserAgent
```

**Summary**

Возвращает UserAgent.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** \YooKassa\Client\UserAgent - 


<a name="method_sendRequest" class="anchor"></a>
#### public sendRequest() : array

```php
public sendRequest() : array
```

**Summary**

Выполняет запрос, получает и возвращает обработанный ответ

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |

**Returns:** array - 


<a name="method_setAdvancedCurlOptions" class="anchor"></a>
#### public setAdvancedCurlOptions() : void

```php
public setAdvancedCurlOptions() : void
```

**Summary**

Устанавливает дополнительные настройки curl.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

**Returns:** void - 


<a name="method_setBearerToken" class="anchor"></a>
#### public setBearerToken() : $this

```php
public setBearerToken(null|string $bearerToken) : $this
```

**Summary**

Устанавливает OAuth-токен магазина.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | bearerToken  | OAuth-токен магазина |

**Returns:** $this - 


<a name="method_setBody" class="anchor"></a>
#### public setBody() : void

```php
public setBody(string $method, string|null $httpBody = null) : void
```

**Summary**

Устанавливает тело запроса.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | method  | HTTP метод |
| <code lang="php">string OR null</code> | httpBody  | Тело запроса |

**Returns:** void - 


<a name="method_setConfig" class="anchor"></a>
#### public setConfig() : void

```php
public setConfig(array $config) : void
```

**Summary**

Устанавливает настройки.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | config  | Настройки клиента |

**Returns:** void - 


<a name="method_setConnectionTimeout" class="anchor"></a>
#### public setConnectionTimeout() : void

```php
public setConnectionTimeout(int $connectionTimeout = 30) : void
```

**Summary**

Устанавливает значение параметра CURLOPT_CONNECTTIMEOUT.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | connectionTimeout  | Число секунд ожидания при попытке подключения |

**Returns:** void - 


<a name="method_setCurlOption" class="anchor"></a>
#### public setCurlOption() : bool

```php
public setCurlOption(string $optionName, mixed $optionValue) : bool
```

**Summary**

Устанавливает параметры CURL.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | optionName  | Имя параметра |
| <code lang="php">mixed</code> | optionValue  | Значение параметра |

**Returns:** bool - 


<a name="method_setKeepAlive" class="anchor"></a>
#### public setKeepAlive() : $this

```php
public setKeepAlive(bool $keepAlive) : $this
```

**Summary**

Устанавливает флаг сохранения соединения.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | keepAlive  | Флаг сохранения настроек |

**Returns:** $this - 


<a name="method_setLogger" class="anchor"></a>
#### public setLogger() : void

```php
public setLogger(?\Psr\Log\LoggerInterface $logger) : void
```

**Summary**

Устанавливает объект для логирования.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?\Psr\Log\LoggerInterface</code> | logger  | Объект для логирования |

**Returns:** void - 


<a name="method_setProxy" class="anchor"></a>
#### public setProxy() : void

```php
public setProxy(string $proxy) : void
```

**Summary**

Устанавливает настройки прокси.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | proxy  | Прокси сервер |

**Returns:** void - 


<a name="method_setShopId" class="anchor"></a>
#### public setShopId() : $this

```php
public setShopId(mixed $shopId) : $this
```

**Summary**

Устанавливает shopId магазина.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | shopId  | shopId магазина |

**Returns:** $this - 


<a name="method_setShopPassword" class="anchor"></a>
#### public setShopPassword() : $this

```php
public setShopPassword(string|null $shopPassword) : $this
```

**Summary**

Устанавливает секретный ключ магазина.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | shopPassword  | Секретный ключ магазина |

**Returns:** $this - 


<a name="method_setTimeout" class="anchor"></a>
#### public setTimeout() : void

```php
public setTimeout(int $timeout) : void
```

**Summary**

Устанавливает значение параметра CURLOPT_TIMEOUT.

**Details:**
* Inherited From: [\YooKassa\Client\CurlClient](../classes/YooKassa-Client-CurlClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | timeout  | Максимальное количество секунд для выполнения функций cURL |

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