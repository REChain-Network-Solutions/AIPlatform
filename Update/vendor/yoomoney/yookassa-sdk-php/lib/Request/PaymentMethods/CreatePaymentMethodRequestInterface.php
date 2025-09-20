<?php

namespace YooKassa\Request\PaymentMethods;

use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation;

/**
 * Класс, представляющий модель CreatePaymentMethodRequest.
 *
 * Данные для проверки и сохранения способа оплаты.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
 * @property PaymentMethodCard $card Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне).
 * @property PaymentMethodHolder $holder Данные магазина, для которого сохраняется способ оплаты.
 * @property string $client_ip IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
 * @property string $clientIp IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
 * @property AbstractConfirmation $confirmation Данные, необходимые для инициирования сценария подтверждения привязки.
 */
interface CreatePaymentMethodRequestInterface extends AbstractRequestInterface
{
    /**
     * Возвращает type.
     *
     * @return string|null
     */
    public function getType(): ?string;

    /**
     * Устанавливает type.
     *
     * @param string|null $type Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
     *
     * @return CreatePaymentMethodRequest
     */
    public function setType(?string $type = null): CreatePaymentMethodRequest;

    /**
     * Возвращает card.
     *
     * @return PaymentMethodCard|null
     */
    public function getCard(): ?PaymentMethodCard;

    /**
     * Устанавливает card.
     *
     * @param PaymentMethodCard|array|null $card Данные для проверки и сохранения банковской карты.
     *
     * @return CreatePaymentMethodRequest
     */
    public function setCard(mixed $card): CreatePaymentMethodRequest;

    /**
     * Возвращает holder.
     *
     * @return PaymentMethodHolder|null
     */
    public function getHolder(): ?PaymentMethodHolder;

    /**
     * Устанавливает holder.
     *
     * @param PaymentMethodHolder|array|null $holder Данные магазина, для которого сохраняется способ оплаты.
     *
     * @return CreatePaymentMethodRequest
     */
    public function setHolder(mixed $holder = null): CreatePaymentMethodRequest;

    /**
     * Возвращает client_ip.
     *
     * @return string|null
     */
    public function getClientIp(): ?string;

    /**
     * Устанавливает client_ip.
     *
     * @param string|null $client_ip IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
     *
     * @return CreatePaymentMethodRequest
     */
    public function setClientIp(?string $client_ip = null): CreatePaymentMethodRequest;

    /**
     * Возвращает confirmation.
     *
     * @return AbstractConfirmation|null
     */
    public function getConfirmation(): ?AbstractConfirmation;

    /**
     * Устанавливает confirmation.
     *
     * @param AbstractConfirmation|array|null $confirmation Данные, необходимые для инициирования сценария подтверждения привязки.
     *
     * @return CreatePaymentMethodRequest
     */
    public function setConfirmation(mixed $confirmation = null): CreatePaymentMethodRequest;
}
