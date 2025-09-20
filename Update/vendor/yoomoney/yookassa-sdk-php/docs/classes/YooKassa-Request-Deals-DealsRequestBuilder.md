# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Deals\DealsRequestBuilder
### Namespace: [\YooKassa\Request\Deals](../namespaces/yookassa-request-deals.md)
---
**Summary:**

Класс, представляющий модель DealsRequestBuilder.

**Description:**

Класс билдера запросов к API для получения списка сделок магазина.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$currentObject](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#property_currentObject) |  | Инстанс собираемого запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [build()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_build) |  | Собирает и возвращает объект запроса списка сделок магазина. |
| public | [setCreatedAtGt()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setCreatedAtGt) |  | Устанавливает дату создания от которой выбираются платежи. |
| public | [setCreatedAtGte()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setCreatedAtGte) |  | Устанавливает дату создания от которой выбираются платежи. |
| public | [setCreatedAtLt()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setCreatedAtLt) |  | Устанавливает дату создания до которой выбираются платежи. |
| public | [setCreatedAtLte()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setCreatedAtLte) |  | Устанавливает дату создания до которой выбираются платежи. |
| public | [setCursor()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setCursor) |  | Устанавливает страница выдачи результатов. |
| public | [setExpiresAtGt()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setExpiresAtGt) |  | Устанавливает дату автоматического закрытия от которой выбираются платежи. |
| public | [setExpiresAtGte()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setExpiresAtGte) |  | Устанавливает дату автоматического закрытия от которой выбираются платежи. |
| public | [setExpiresAtLt()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setExpiresAtLt) |  | Устанавливает дату автоматического закрытия до которой выбираются платежи. |
| public | [setExpiresAtLte()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setExpiresAtLte) |  | Устанавливает дату автоматического закрытия до которой выбираются платежи. |
| public | [setFullTextSearch()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setFullTextSearch) |  | Устанавливает фильтр по описанию выбираемых сделок. |
| public | [setLimit()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setLimit) |  | Устанавливает ограничение количества объектов сделки. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setStatus()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_setStatus) |  | Устанавливает статус выбираемых сделок. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md#method_initCurrentObject) |  | Возвращает новый объект запроса для получения списка сделок, который в дальнейшем будет собираться в билдере. |

---
### Details
* File: [lib/Request/Deals/DealsRequestBuilder.php](../../lib/Request/Deals/DealsRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\Deals\DealsRequestBuilder

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

Инстанс собираемого запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>
Собираемый объект запроса списка сделок магазина
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
#### public build() : \YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Common\AbstractRequestInterface
```

**Summary**

Собирает и возвращает объект запроса списка сделок магазина.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив с настройками запроса |

**Returns:** \YooKassa\Common\AbstractRequestInterface - Инстанс объекта запроса к API для получения списка сделок магазина


<a name="method_setCreatedAtGt" class="anchor"></a>
#### public setCreatedAtGt() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setCreatedAtGt(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату создания от которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время создания, от (не включая) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setCreatedAtGte" class="anchor"></a>
#### public setCreatedAtGte() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setCreatedAtGte(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату создания от которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время создания, от (включительно) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setCreatedAtLt" class="anchor"></a>
#### public setCreatedAtLt() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setCreatedAtLt(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату создания до которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время создания, до (не включая) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setCreatedAtLte" class="anchor"></a>
#### public setCreatedAtLte() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setCreatedAtLte(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату создания до которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время создания, до (включительно) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setCursor" class="anchor"></a>
#### public setCursor() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setCursor(null|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает страница выдачи результатов.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | value  | Страница выдачи результатов или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setExpiresAtGt" class="anchor"></a>
#### public setExpiresAtGt() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setExpiresAtGt(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату автоматического закрытия от которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время автоматического закрытия, до (включительно) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setExpiresAtGte" class="anchor"></a>
#### public setExpiresAtGte() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setExpiresAtGte(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату автоматического закрытия от которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время автоматического закрытия, от (включительно) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setExpiresAtLt" class="anchor"></a>
#### public setExpiresAtLt() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setExpiresAtLt(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату автоматического закрытия до которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время автоматического закрытия, до (включительно) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setExpiresAtLte" class="anchor"></a>
#### public setExpiresAtLte() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setExpiresAtLte(null|\DateTime|int|string $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает дату автоматического закрытия до которой выбираются платежи.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \DateTime OR int OR string</code> | value  | Время автоматического закрытия, до (включительно) или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setFullTextSearch" class="anchor"></a>
#### public setFullTextSearch() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setFullTextSearch(string|null $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает фильтр по описанию выбираемых сделок.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Фильтр по описанию выбираемых сделок или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_setLimit" class="anchor"></a>
#### public setLimit() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setLimit(null|string|int $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает ограничение количества объектов сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string OR int</code> | value  | Ограничение количества объектов сделки или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


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


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : \YooKassa\Request\Deals\DealsRequestBuilder

```php
public setStatus(string|null $value) : \YooKassa\Request\Deals\DealsRequestBuilder
```

**Summary**

Устанавливает статус выбираемых сделок.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Статус выбираемых сделок или null, чтобы удалить значение |

**Returns:** \YooKassa\Request\Deals\DealsRequestBuilder - Инстанс текущего билдера


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\Deals\DealsRequest

```php
protected initCurrentObject() : \YooKassa\Request\Deals\DealsRequest
```

**Summary**

Возвращает новый объект запроса для получения списка сделок, который в дальнейшем будет собираться в билдере.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestBuilder](../classes/YooKassa-Request-Deals-DealsRequestBuilder.md)

**Returns:** \YooKassa\Request\Deals\DealsRequest - Объект запроса списка сделок магазина



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