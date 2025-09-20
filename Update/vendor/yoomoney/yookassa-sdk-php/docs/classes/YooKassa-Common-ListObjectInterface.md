# [YooKassa API SDK](../home.md)

# Interface: ListObjectInterface
### Namespace: [\YooKassa\Common](../namespaces/yookassa-common.md)
---
**Summary:**

Interface ListObjectInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [add()](../classes/YooKassa-Common-ListObjectInterface.md#method_add) |  |  |
| public | [clear()](../classes/YooKassa-Common-ListObjectInterface.md#method_clear) |  |  |
| public | [get()](../classes/YooKassa-Common-ListObjectInterface.md#method_get) |  |  |
| public | [getItems()](../classes/YooKassa-Common-ListObjectInterface.md#method_getItems) |  |  |
| public | [getType()](../classes/YooKassa-Common-ListObjectInterface.md#method_getType) |  |  |
| public | [isEmpty()](../classes/YooKassa-Common-ListObjectInterface.md#method_isEmpty) |  |  |
| public | [merge()](../classes/YooKassa-Common-ListObjectInterface.md#method_merge) |  |  |
| public | [remove()](../classes/YooKassa-Common-ListObjectInterface.md#method_remove) |  |  |
| public | [setType()](../classes/YooKassa-Common-ListObjectInterface.md#method_setType) |  |  |
| public | [toArray()](../classes/YooKassa-Common-ListObjectInterface.md#method_toArray) |  |  |

---
### Details
* File: [lib/Common/ListObjectInterface.php](../../lib/Common/ListObjectInterface.php)
* Package: \YooKassa
* Parents:
  * [](\ArrayAccess)
  * [](\JsonSerializable)
  * [](\Countable)
  * [](\IteratorAggregate)
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
#### public getType() : string

```php
public getType() : string
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

**Returns:** string - 


<a name="method_setType" class="anchor"></a>
#### public setType() : $this

```php
public setType(string $type) : $this
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  |  |

**Returns:** $this - 


<a name="method_add" class="anchor"></a>
#### public add() : $this

```php
public add(mixed $item) : $this
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | item  |  |

**Returns:** $this - 


<a name="method_merge" class="anchor"></a>
#### public merge() : $this

```php
public merge(iterable $data) : $this
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable</code> | data  |  |

**Returns:** $this - 


<a name="method_remove" class="anchor"></a>
#### public remove() : $this

```php
public remove(int $index) : $this
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | index  |  |

**Returns:** $this - 


<a name="method_clear" class="anchor"></a>
#### public clear() : $this

```php
public clear() : $this
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

**Returns:** $this - 


<a name="method_isEmpty" class="anchor"></a>
#### public isEmpty() : bool

```php
public isEmpty() : bool
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

**Returns:** bool - 


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \Ds\Vector

```php
public getItems() : \Ds\Vector
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

**Returns:** \Ds\Vector - 


<a name="method_get" class="anchor"></a>
#### public get() : \YooKassa\Common\AbstractObject

```php
public get(int $index) : \YooKassa\Common\AbstractObject
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | index  |  |

**Returns:** \YooKassa\Common\AbstractObject - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

**Returns:** array - 




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