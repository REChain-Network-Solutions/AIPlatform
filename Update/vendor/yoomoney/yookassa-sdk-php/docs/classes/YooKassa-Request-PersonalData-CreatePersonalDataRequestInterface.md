# [YooKassa API SDK](../home.md)

# Interface: CreatePersonalDataRequestInterface
### Namespace: [\YooKassa\Request\PersonalData](../namespaces/yookassa-request-personaldata.md)
---
**Summary:**

Interface CreatePersonalDataRequestInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getFirstName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_getFirstName) |  | Возвращает имя пользователя. |
| public | [getLastName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_getLastName) |  | Возвращает фамилию пользователя. |
| public | [getMetadata()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_getMetadata) |  | Возвращает метаданные. |
| public | [getMiddleName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_getMiddleName) |  | Возвращает отчество пользователя. |
| public | [getType()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_getType) |  | Возвращает тип персональных данных. |
| public | [hasFirstName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_hasFirstName) |  | Проверяет наличие имени пользователя в запросе. |
| public | [hasLastName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_hasLastName) |  | Проверяет наличие фамилии пользователя в запросе. |
| public | [hasMetadata()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные. |
| public | [hasMiddleName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_hasMiddleName) |  | Проверяет наличие отчества пользователя в запросе. |
| public | [hasType()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_hasType) |  | Проверяет наличие типа персональных данных в запросе. |
| public | [setFirstName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_setFirstName) |  | Устанавливает имя пользователя. |
| public | [setLastName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_setLastName) |  | Устанавливает фамилию пользователя. |
| public | [setMetadata()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_setMetadata) |  | Устанавливает метаданные. |
| public | [setMiddleName()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_setMiddleName) |  | Устанавливает отчество пользователя. |
| public | [setType()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_setType) |  | Устанавливает тип персональных данных. |
| public | [toArray()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [validate()](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md#method_validate) |  | Проверяет на валидность текущий объект |

---
### Details
* File: [lib/Request/PersonalData/CreatePersonalDataRequestInterface.php](../../lib/Request/PersonalData/CreatePersonalDataRequestInterface.php)
* Package: \YooKassa\Request
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
<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип персональных данных.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** string|null - Тип персональных данных


<a name="method_setType" class="anchor"></a>
#### public setType() : $this

```php
public setType(string|null $type) : $this
```

**Summary**

Устанавливает тип персональных данных.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип персональных данных |

**Returns:** $this - 


<a name="method_hasType" class="anchor"></a>
#### public hasType() : bool

```php
public hasType() : bool
```

**Summary**

Проверяет наличие типа персональных данных в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** bool - True если тип персональных данных задан, false если нет


<a name="method_getLastName" class="anchor"></a>
#### public getLastName() : string|null

```php
public getLastName() : string|null
```

**Summary**

Возвращает фамилию пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** string|null - Фамилия пользователя


<a name="method_setLastName" class="anchor"></a>
#### public setLastName() : $this

```php
public setLastName(string|null $last_name) : $this
```

**Summary**

Устанавливает фамилию пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | last_name  | Фамилия пользователя |

**Returns:** $this - 


<a name="method_hasLastName" class="anchor"></a>
#### public hasLastName() : bool

```php
public hasLastName() : bool
```

**Summary**

Проверяет наличие фамилии пользователя в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** bool - True если фамилия пользователя задана, false если нет


<a name="method_getFirstName" class="anchor"></a>
#### public getFirstName() : string|null

```php
public getFirstName() : string|null
```

**Summary**

Возвращает имя пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** string|null - Имя пользователя


<a name="method_setFirstName" class="anchor"></a>
#### public setFirstName() : $this

```php
public setFirstName(string|null $first_name) : $this
```

**Summary**

Устанавливает имя пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | first_name  | Имя пользователя |

**Returns:** $this - 


<a name="method_hasFirstName" class="anchor"></a>
#### public hasFirstName() : bool

```php
public hasFirstName() : bool
```

**Summary**

Проверяет наличие имени пользователя в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** bool - True если имя пользователя задано, false если нет


<a name="method_getMiddleName" class="anchor"></a>
#### public getMiddleName() : null|string

```php
public getMiddleName() : null|string
```

**Summary**

Возвращает отчество пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** null|string - Отчество пользователя


<a name="method_setMiddleName" class="anchor"></a>
#### public setMiddleName() : $this

```php
public setMiddleName(null|string $middle_name = null) : $this
```

**Summary**

Устанавливает отчество пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | middle_name  | Отчество пользователя |

**Returns:** $this - 


<a name="method_hasMiddleName" class="anchor"></a>
#### public hasMiddleName() : bool

```php
public hasMiddleName() : bool
```

**Summary**

Проверяет наличие отчества пользователя в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** bool - True если отчество пользователя задано, false если нет


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает метаданные.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : $this

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $metadata) : $this
```

**Summary**

Устанавливает метаданные.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | metadata  | Метаданные |

**Returns:** $this - 


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Проверяет на валидность текущий объект

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** bool - True если объект запроса валиден, false если нет


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePersonalDataRequestInterface](../classes/YooKassa-Request-PersonalData-CreatePersonalDataRequestInterface.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта




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