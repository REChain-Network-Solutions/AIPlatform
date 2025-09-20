# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder
### Namespace: [\YooKassa\Request\SelfEmployed](../namespaces/yookassa-request-selfemployed.md)
---
**Summary:**

Класс, представляющий модель SelfEmployedRequestBuilder.

**Description:**

Класс билдера объектов запросов к API на создание самозанятого.

---
### Examples
Пример использования билдера

```php
try {
    $selfEmployedBuilder = \YooKassa\Request\SelfEmployed\SelfEmployedRequest::builder();
    $selfEmployedBuilder
        ->setItn('123456789012')
        ->setPhone('79001002030')
        ->setConfirmation(['type' => \YooKassa\Model\SelfEmployed\SelfEmployedConfirmationType::REDIRECT])
    ;

    // Создаем объект запроса
    $request = $selfEmployedBuilder->build();

    $idempotenceKey = uniqid('', true);
    $response = $client->createSelfEmployed($request, $idempotenceKey);
} catch (Exception $e) {
    $response = $e;
}

var_dump($response);

```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$currentObject](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [build()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setConfirmation()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md#method_setConfirmation) |  | Устанавливает сценарий подтверждения. |
| public | [setItn()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md#method_setItn) |  | Устанавливает ИНН самозанятого. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setPhone()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md#method_setPhone) |  | Устанавливает телефон самозанятого. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/SelfEmployed/SelfEmployedRequestBuilder.php](../../lib/Request/SelfEmployed/SelfEmployedRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct() : mixed
```

**Summary**

Конструктор, инициализирует пустой запрос, который в будущем начнём собирать.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

**Returns:** mixed - 


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface|\YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface|\YooKassa\Common\AbstractRequestInterface
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidRequestException | Выбрасывается если собрать объект запроса не удалось |

**Returns:** \YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface|\YooKassa\Common\AbstractRequestInterface - Инстанс объекта запроса


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : self

```php
public setConfirmation(null|array|\YooKassa\Model\SelfEmployed\SelfEmployedConfirmation $value) : self
```

**Summary**

Устанавливает сценарий подтверждения.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation</code> | value  | сценарий подтверждения |

**Returns:** self - Инстанс билдера запросов


<a name="method_setItn" class="anchor"></a>
#### public setItn() : self

```php
public setItn(null|string $value) : self
```

**Summary**

Устанавливает ИНН самозанятого.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | value  | ИНН самозанятого |

**Returns:** self - Инстанс билдера запросов


<a name="method_setOptions" class="anchor"></a>
#### public setOptions() : \YooKassa\Common\AbstractRequestBuilder

```php
public setOptions(iterable|null $options) : \YooKassa\Common\AbstractRequestBuilder
```

**Summary**

Устанавливает свойства запроса из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable OR null</code> | options  | Массив свойств запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \InvalidArgumentException | Выбрасывается если аргумент не массив и не итерируемый объект |
| \YooKassa\Common\Exceptions\InvalidPropertyException | Выбрасывается если не удалось установить один из параметров, переданных в массиве настроек |

**Returns:** \YooKassa\Common\AbstractRequestBuilder - Инстанс текущего билдера запросов


<a name="method_setPhone" class="anchor"></a>
#### public setPhone() : self

```php
public setPhone(null|string $value) : self
```

**Summary**

Устанавливает телефон самозанятого.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | value  | телефон самозанятого |

**Returns:** self - Инстанс билдера запросов


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\SelfEmployed\SelfEmployedRequest

```php
protected initCurrentObject() : \YooKassa\Request\SelfEmployed\SelfEmployedRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestBuilder](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestBuilder.md)

**Returns:** \YooKassa\Request\SelfEmployed\SelfEmployedRequest - Инстанс собираемого объекта запроса к API



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