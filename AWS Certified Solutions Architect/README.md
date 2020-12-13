# Designing Resilient Architectures
## Axioms
- "Single AZ" will never be a right answer
- Using AWS managed services should always be preferred
- Fault tolerant and high availability are not the same thing
- Expect failure and design accordingly

## Most important patterns:
- Multi-tier
- Serverless
- Microservices

## High availability vs. fault-tolerance
**FT is a higher bar than HA**. HA means the system is up and available but it might perform in a degraded state. FT means the user does not experience any impact from a fault.

*SLA* - service level agreement

# Designing Performant Architectures
## Axioms
- Use S3 for unstructured data
- Use caching to improve performance
- Use Auto Scaling

## Performant Storage
- Use Redshift for analytic queries and RDS for transactional queries
- Use RDS for transactions or complex queries
- Use DynamoDB for massive read/write rates (e.g. 150K write/s)
- Use DynamoDB for dynamic sharding

## Caching
- CloudFront
- ElastiCache

# Designing Secure Applications
## Axioms
- Lock down the root user
- Security groups are stateful and only *allow*. Network ACLs are stateless and allow for explicit *deny*
- Prefer IAM Roles to access keys 

# Designing Cost-optimized architectures
## Axioms
- If you know it's going to be on, reserve it
- Any unused CPU time is a waste of money
- Use the most cost-effective data storage service and class
- Determine the most cost-effective EC2 pricing model and instance type

> *When you use AWS you can control your spend following a simple set of steps. 1) Right-size your services to meet capacity needs at the lowest cost 2) Save when you reserve 3) Use the spot market 4) Monitor and track service usage 5) Use Cost Explorer to optimize savings. Scripts (Lambda) can be used to stop idle instances to save cost. A properly configured auto scaling group will perform the same function but scripts are useful for instances not in an auto scaling group. One of the benefits of AWS is that you can quickly and cheaply spin up dev and test environments which match the exact specifications of your production environment. Cloud Formation can quickly spin up these environments which makes it beneficial to quickly remove the environment when not in use to save cost.*

# Defining Operationally-excellent Architectures
*The ability to run and monitor systems to deliver business value and continually improve supporting processes and procedures: **Prepare, Operate, Evolve***

## Axioms
- IAM roles are easier and safer than keys and passwords
- Monitor metrics across the system
- Automate responses to metrics where appropriate
- Provide alerts for anomalous conditions

# Backup and Disaster Recovery Strategies
Strategy considerations include defining Recovery Time Objective (RTO) and
Recovery Point Objective (RPO).

- Backup & Restore
Using AWS as a virtual tape library. Uses; AWS Storage gateway, AWS import/export, Glacier/S3

- Pilot Light
Minimal version of environment running on AWS which can be lit up and expanded to full size.	Uses: AMI's boot strapping, EIPs, ELBs, CloudFormation, RDS replication

- Warm stand by
Scaled down version of fully functional environment always running. Uses services as in *Pilot Light*

- Multi site
Fully operational version of an environment running off site or in another region. Uses all AWS services

# Storage
## S3
### Security
There are four ways to control access to resources:
- IAM Policies - allow/restrict access to buckets/objects,
- Bucket Policies - JSON policies assigned to buckets,
- Access Control Lists (ACLs) - which user or AWS account can access a bucket or object using range of permissions, i.e. read/write,
- Query String Authentication

Bucket policies provide centralized, access control to buckets and objects based on a variety of conditions, including Amazon S3 operations, requesters, resources, and aspects of the request (e.g., IP address). If a user needs to apply a policy on individual objects, he should use the access control lists, which can add (grant) permissions only on individual objects, policies can either add or deny permissions across all (or a subset) of objects within a bucket. With one request an account can set the permissions of any number of objects in a bucket.

The recommended ways to provide access are IAM policies and bucket policies (and not ACLs).

Other  features:
- Lifecycle Policies - move data between storage classes, e.g. Standard - Infrequent Access (IA),
- MFA Delete - User has to enter a 6 digit MFA Code to delete an object,
- Versioning
- Encryption: SSE-S3, SSE-KMS, SSE-C

Batch operation can automate: `PUT object tagging` and `PUT copy object` operations only.

### Object locks
Object locks can prevent objects from modification or deletion
- Users can only enable Object locks can on new S3 buckets. 
- Object locks have two retention modes - governance mode and compliance mode.
	- Compliance mode prevents objects from being deleted or updated by users, including the root user. 
	- Governance mode allows objects to be modified, but prevents objects from being deleted.
- Legal holds have no specific time frame associated with them, and cannot be configured at a bucket level. They must be configured at an object version level.

### Costs
#### Data Transfer
Data transfer is free when:
- From Internet to S3,
- From S3 to EC2 in the same region
- From S3 to CloudFront

### Consistency model
S3 is strongly consistent for new objects upload and eventually consistent for updates (overwirte PUTS and DELETES)

### Notes
- The bucket policy applies only to objects that are owned by the bucket owner. If your bucket contains objects that aren't owned by the bucket owner, public READ permission on those objects should be granted using the object access control list (ACL).

## EBS
- Data stored on EBS is automatically replicated within an AZ (not across multiple zones)

## Glacier
There are three retrieval options:
- Expedited (few minutes)
- Standard (few hours)
- Bulk (up to 12h)

## Storage Gateway
AWS Storage Gateway connects an on-premises software appliance with cloud-based storage to provide seamless integration with data security features between your on-premises IT environment and the Amazon Web Services (AWS) storage infrastructure.

Configurations possible:

- File gateways (NFS + S3) - *allows to map/mount drives to S3 bucket as if it was local. Presented as a NFS share*
- Volume gateways (iSCSI + S3) - *Volume gateways are stored on block storage volumes, similar to Amazon EBS. The volumes offer a limited amount of storage, even though the volumes can be stored in Amazon S3*
	- Stored volume (Store on Local + Async S3 backup as EBS snapshots)
	- Cached volume (Cached on Local + Store on S3)
- Tape gateways (Stores virtual tapes in a virtual tape library VTL backed by S3, leverages Glacier for data archiving)

General Data Storage systems:
- NAS - Network-Attached Storage (Network-attached storage offers dedicated file serving and sharing through a network)
- DAS - Direct-Attached Storage (Storage system is a part of the actual host computer or connected directly to the host computer)
- SAN - Storage Area Network (A storage area network is a dedicated, high-performance storage system that transfers block-level data between servers and storage devices. SAN is typically used in data centers, enterprises or virtual computing environments. It offers the speed of DAS with the sharing, flexibility and reliability of NAS)

## Notes
- EFS can be shared between multiple instances (even across several regions), whereas EBS is a block storage that needs to be attached to an instance and cannot be used by multiple instances.
*EDIT (as of 2020): 
Amazon EBS Multi-Attach enables you to attach a single Provisioned IOPS SSD (io1) volume to up to 16 Nitro-based instances that are in the same Availability Zone.*

# Security
SCP - Service Control Policies - determine maximum level of permissions for a given AWS account (from AWS Organisations), i.e. they add a guardrail to define what is allowed. The following are not affected by SCPs: Actions performed by the master account, service-linked roles, managing CloudFront keys

### WAF
WAF components are: Conditions, Rules, Web ACLs. Rule is a set of conditions and in order to be met, ALL conditions need to be met. Within the ACL an action is applied to each rule - it could be Allow, Block or Count.

#### OWASP TOP10 Vulnerabilities
- Injections
- Broken Authentication and Session Management
- XSS
- Insecure Direct Object References
- Security Misconfiguration
- Sensitive Data Exposure
- Missing Function level access control
- CSRF
- Using known vulnerable components
- Unvalidated reads and forwards

### Firewall Manager
Helps in managing WAF in a multi-account environment. Rule groups are group of WAF rules that have the same action when the conditions are met.

### Shield
Protects the infrastructure againts the DDOS attacks. Common DDOS attacks include:
- SYN flood (sending spoofed SYN packets but not responding to SYN/ACK)
- DNS query flood
- HTTP flood/Cache-busting

Shield comes in two versions: Standard (operating at network and transport layer) and Advanced (operating at network, transport and application layers).

## Cognito
- **Identity pools** provide temporary access to defined AWS services to your application
users
- Sign-up and sign-in is managed through **Cognito user pools**
- Cognito User Pools are used for authentication whereas Identity Pools are used for authorization

# Networking
## VPC - Virtual Private Cloud
> Isolated Segment of the AWS infrastructure allowing to provision cloud resources

You can create up to 5 VPCs per AWS account per region. To to that you need to specify a *name* and a *CIDR* (Classless Inter-Domain Routing). You cannot change the CIDR of your VPC once created

You may connect your Amazon VPC to:

- The internet (via an internet gateway)
- Your corporate data center using an AWS Site-to-Site VPN connection (via the virtual private gateway)
- Both the internet and your corporate data center (utilizing both an internet gateway and a virtual private gateway)
- Other AWS services (via internet gateway, NAT, virtual private gateway, or VPC endpoints)
- Other Amazon VPCs (via VPC peering connections)

### VPC Endpoints
*VPC endpoints enable you to privately connect your VPC to services hosted on AWS without requiring an Internet gateway, a NAT device, VPN, or firewall proxies. Endpoints are horizontally scalable and highly available virtual devices that allow communication between instances in your VPC and AWS services. Amazon VPC offers two different types of endpoints: 1) gateway type endpoints (S3 and DynamoDB) and 2) interface type endpoints.*

*You can create your own application in your VPC and configure it as an AWS PrivateLink-powered service (referred to as an endpoint service). Other AWS principals can create a connection from their VPC to your endpoint service using an interface VPC endpoint. You are the service provider, and the AWS principals that create connections to your service are service consumers*

## Subnets
- Subnets reside within a VPC and allow to segment a VPC into multiple different networks.
- Allowed subnet masks ranges are from /16 up to /28
- If a subnet doesn't have a route to the Internet gateway, but has its traffic routed to a virtual private gateway for a VPN connection, the subnet is known as a VPN-only subnet.

### Route table
In order to access internet from the VPC you need to attach an **Internet Gateway** to the VPC. In order for the **Subnet** to be able to access internet you need to add a route to a subnet's route table. 

Same route table can be assigned to multiple subnets but you cannot have multiple route tables assigned to a single subnet.

## Security
### Network Access Control Lists (NACLs)
NACLs can define both ingress and egress traffic (both allow and deny) - there is also a default rule that denies all inbound/outbound traffic into/from a subnet. A network ACL is a numbered list of rules that are evaluated in order, starting with the lowest numbered rule, to determine whether traffic is allowed in or out of any subnet associated with the network ACL.

- NACLs are stateless

### Security Groups
They are like NACLs but on the instance (not subnet) level.
Unlike in NACLs the rules are only whitelists - if there is no rule then the traffic is dropped.

- Security Groups are stateful (you don't need to configure ruls for the return traffic)

### NAT Gateway
If a private subnet needs an outbound access to the internet (e.g. to download OS patches), we need to create a public subnet with a **NAT gateway** and configure private subnet's route table to point to the NAT gateway. Note that the NAT gateway will block any inbound traffic coming from the internet.

### NAT Instance
It's an EC2 instance with a preconfigured Linux AMI. You must disable the source/destination check on the NAT instance’s ENI (Elastic Network Interface) which allows the NAT instance to receive traffic addressed to an IP other than its own, and it also allows the instance to send traffic using a source IP that it doesn’t own. 
- You can use a NAT instance as a bastion host

## VPC Connectivity
### VPN
If you want to connect your VPC to a remote data center you need to set up a **Virtual Gateway (VGW)** in the VPC (or **VPN Gateway (VPG)**) and a **Customer Gateway (CGW)** in the data center.

### Direct Connect
In DC you don't use internet but connect to AWS through a DC location which is a separate physical facility. You'll also need a **Virtual Gateway** attached to your AWS VPC.

There are two types of virtual interface connections: public and private. Virtual public interfaces can connect a remote data center to AWS regional services in a specific region. Virtual private interfaces can connect a remote data center to resources within a specific VPC.

### VPC Peering
Allows two VPCs to talk to each other (it's a 1-1 connection). Their CIDR blocks cannot overlap.

### Transit Gateway (TGW)
Allows for VPC peering with 1-many connections. You can connect to the Transit Gateway both from the AWS or from the datacenter via VPN or Direct Connect.

## VPC Endpoints
Using VPC Endpoints you can connect to AWS services using AWS internal network instead of internet, without a need to configure DirectConnect, NAT gateways etc.
There are two types of VPC endpoints:
- Interface endpoints (uses Privatelink)
- Gateway endpoints (S3 and DynamoDB)

## AWS Global Accelerator
AWS Global Accelerator provides static IP addresses that act as a fixed entry point to your application endpoints in a single or multiple AWS Regions and uses the AWS global network to optimise the path from the users to the applcation.

## Elastic Network Interface
An elastic network interface is a logical networking component in a VPC that represents a virtual network card. The default one is `eth0`.

- The ENI allows the user to change the security group of a running instance

## Route 53
There are 7 routing policies:
- Simple (hardcoded IP)
- Failover (based on the health of resources)
- Geo-Location (geographic location)
- Geoproximity (based on location of both users and resources)
- Latency
- Multivalue Answer (random)
- Wighted (based on proportions)

### Failover configuartions
- Active-active failover: Use this failover configuration when you want all of your resources to be available the majority of the time. Can be used only with failover routing policy
- Active-passive failover: Use this failover configuration when you want a primary group of resources to be available the majority of the time and you want a secondary group of resources to be on standby in case all of the primary resources become unavailable. Can be used with any routing policy other than failover
- Active-active-passive and other mixed configurations: You can combine alias and non-alias resource record sets to produce a variety of Amazon Route 53 behaviors

### Notes
- A health check can monitor the status of other health checks; this type of health check is known as a calculated health check. The health check that does the monitoring is the parent health check, and the health checks that are monitored are child health checks. 

## Multi-tier
Multi tier refers to having three tiers of an application:
**Presentation, Logic (Application), Data (Database)**
independent of each other, so that each of them can be scaled separately.

Typical sofware stack patterns:
- LAMP: Linux + Apache + MySQL + PHP
- MEAN: MongoDB + ExpressJS + Angular JS + NodeJS
- Serverless: AWS API Gateway + AWS Lambda

## Load Balancing
There are three types of ELBs:
- Network Load Balancer
Manages TCP/UDP/TLS, operates at connection level (layer 4 OSI), should **ideally sit between the Internet Gateway and the presentation tier**, manages SSL certificates, can be used to manage security (e.g. protect from DOS)
- Application Load Balancer
Operates at request level (layer 7 OSI), best suited for HTTP/HTTPS, should **ideally sit between the front-end tier and the logic tier** (allows for scaling logic tier easily)
- Classic Load Balancer
Used for EC2 classic infrastructure (deprecated)

### Scaling policy types
Amazon EC2 Auto Scaling supports the following types of scaling policies:

- Target tracking scaling — Increase or decrease the current capacity of the group based on a target value for a specific metric. This is similar to the way that your thermostat maintains the temperature of your home — you select a temperature and the thermostat does the rest.

- Step scaling — Increase or decrease the current capacity of the group based on a set of scaling adjustments, known as step adjustments, that vary based on the size of the alarm breach.

- Simple scaling — Increase or decrease the current capacity of the group based on a single scaling adjustment.

*If you are scaling based on a utilization metric that increases or decreases proportionally to the number of instances in an Auto Scaling group, we recommend that you use target tracking scaling policies. Otherwise, we recommend that you use step scaling policies.*

## Serverless Design Pattern
- You can deploy a DB tier in a private subnet and configure AWS Lambda to access the database through **Elastic Network Interface**
- In multi-tier design usually one lambda is responsible for dealing one API (i.e. one public method), and the delegation is made by the **API Gateway**

### Database
Databse performance can be improved by introducing cache, e.g. **AWS ElastiCache**

## Decoupled and Event-Driven Architectures
Event-driven architecutres involve: **A Producer, The Event Router and the Consumer**. AWS Simple Queue Service (SQS) can be used to build such architecture. 

### SQS Queue types
- Standard Queues (at least once delivery, might not perserve message order, high scalability and unlimited no of transactions)
- FIFO (no duplication of messages, limited no of transactions per sec)
- Dead-Letter (sends messages that fail to be processed). Can be standard or FIFO

### SNS
- SNS is a publish/subscribe messaging service
- SNS is used as a producer for an SQS queue
- SNS can invoke Lambda functions

### Kinesis
There are three variations:
- Amazon Kinesis Streams (Data & Video)
- Amazon Kinesis Data Firehose (load data into data store)
- Amazon Kinesis Analytics (run SQL queries on streaming data)

# Database
Database design considerations:
- CAP (Consistency, Availability, Partition Tolerance),
- Latency
- Durability
- Scalability
- Nondatabase Hosting (e.g. store data on S3 and query using Amazon Redshift Spectrum)

Available engines for AWS RDS:
- MySQL
- MariaDB
- PosgreSQL
- Aurora
- Oracle
- Microsoft Server

Notes:
- All engines excluding Aurora use EBS as storage and Aurora uses Shared Cluster Storage (scales automatically).

- Read replicas are not used for resiliency or availability (only for scaling!). They are available for MySQL, MariaDB and PosgreSQL. Microsoft Server uses Microsoft Server mirroring. RDS MySQL, PostgreSQL, and MariaDB can have up to 5 read replicas, and Amazon Aurora can have up to 15 read replicas.

- PostgreSQL does not allow for nested multi AZ read replicas (i.e. read replica of a read replica).

- RDS provides three storage types: magnetic, General Purpose (SSD), and Provisioned IOPS (input/output operations per second). General purpose is good for small to medium sized databases and has the burst to handle spikes.

- Automated backups are deleted when the DB instance is deleted. Only manually created DB Snapshots are retained after the DB Instance is deleted.

## Services

### Aurora
- MySQL-compatible version of Aurora allows for a **Backtrack** feature that allows to go back in time and recover from an error or incident. 

- Compute and storage are decoupled and can be scaled separately

### DynamoDB
- Provides a secondary level of availability in the form of cross region replication (Global Tables)

- You can create a global secondary index for an existing table at any time. You can
create a local secondary index only when you create the table.

- DAX (DynamoDB Accelerator) is an in-memory cache that allows for response times in microseconds. Note that DAX is a separate entity to DynamoDB and sitts within a VPC (unlike DynamoDB that sits outside of VPC)

### ElastiCache
Elasticache supports:
- ElastiCache for Memchached
- ElastiCache for Redis

#### Components
- Node: fixed chunk of secure, network-attached RAM
- Shard: (Redis shard) - a group of up to 6 ElastiCache nodes
- Redis Cluster - group of 1-90 Redis shards

### Amazon Neptune
Graph database allowing for complex queries (Apache Tinkerpop Gremlin or W3C SPARQL) on highly connected data (graphs). Use cases:
- Social networking
- Fraud detection
- Recommendation engines

### Redshift
Data Warehouse based on PostgreSQL.
Useful not when you want to insert or get a single row (transaction queries) rather when you want to compute aggregate numbers across the entire table (analytic queries)


### Quantum Ledger Database
A managed serverless ledger database (centralised, unlike blockchain). It's immutable and append only - data is placed inside Amazon Ion Documents (superset of JSON).

### DocumentDB
JSON-like database allowing for queries and indexing, compatible with MongoDB.

### Keyspaces
Serverless database service compatible with Apache Cassandra (distributed wide-column store). Queries are performed using Cassandra Query Language (CQL). Keyspace is essentially a group of tables.	

## Costs
### Data Transfer
Data transfer is free when:
- Into RDS from internet
- Out to CloudFront
- Out to EC2 within the same AZ
- Between AZs for multi-az replication

## Key Management Service
- Commonly used symmetric key algorithms: AES (Advanced Encryption Standard), DES (Digital Encryption Standard), Triple-DES and Blowfish.

- Commonly used asymetric algorithms are: RSA (Rivest-Shamir-Adleman), Diffie-Helman, Digital Signature Algorithm

- KMS only provides encryption at rest. In order to encrypt in-transit you need to use e.g. SSL	

- There are two types of CMKs: AWS managed CMKs and Customer managed CMKs

- DEK - Data Encryption Key - is the key used to encrypt data. First, CMK creates a plaintext DEK key and then an encrypted DEK. Plaintext DEK is used to encrypt data and then deleted, and encrypted DEK is stored together with the data. Process of one key encrypting another key is called **envelope encryption**

### Key Policies
They allow to define who can use and access a key in KMS

### Grants
Grants allow you to delegate a subset of your own access to a CMK. They generate a GrantToken and a GrantID

# Auditng, Monitoring and Evaluating
## S3
S3 Event Notifications can be set up on objects stored in S3.Logging is turned off by default but can be enabled. When enabled, they can track requests to your S3 bucket.

S3 Server Access Logging tracks detailed information not currently available with CloudWatch metrics or CloudTrail logs. To track requests for access to your bucket, you can enable access logging. Each access log record provides details about a single access request, such as the requester, bucket name, request time, request action, response status, and error code, if any. Access log information can be useful in security and access audits. It can also help you learn about your customer base and understand your Amazon S3 bill.

## AWS Config
Consists of a Confguration Items that contains configuration information, relationship information and other metadata. CIs are being sent to Configuration Stream when emitted.

# Billing
Billing is a worldwide metric, not specific to any region.
> A user has enabled a CloudWatch alarm. The alarm has a $50 threshold for total AWS charges in the US East (N. Virginia) region. When does CloudWatch trigger the alarm for this threshold?

Answer: When the total AWS charges exceed $50 (NOT When the total charges for the US East (N. Virginia) region exceeds $50)




# Exam notes
## Architecture
- The `AmazonSQSBufferedAsyncClient` for Java provides an implementation of the AmazonSQSAsyncClient interface and adds several important features (NOTE: It doesn't support FIFO queues):
	- Automatic batching of multiple SendMessage, DeleteMessage, or ChangeMessageVisibility requests without any required changes to the application
	- Prefetching of messages into a local buffer that allows your application to immediately process messages from Amazon SQS without waiting for the messages to be retrieved
- You can achieve high availability for a low cost using API Gateway + Lambda as backend
- Streaming -> Kinesis, data for future analysis -> Kinesis Firehouse
- Application running on premises which needs access to S3 buckets assuming there is a VPN connection to AWS -> you need to configure a proxy on Amazon EC2 and use an Amazon S3 VPC endpoint
- You can send fanout (feed to multiple consumers) event notifications with SQS + SNS. Note: SNS does not support SQS FIFO
- You see Disaster Recovery -> think multiple regions. Fault tolerance -> Multi-AZ
- Kinesis:
	- Clicstream analysis case: Kinesis Data Firehouse collects the data and sends it to Kinesis Data Analytics. Kinesis Data Analytics processes data in real time. Kinesis Data Firehose loads processed data into Redshift.
	- Analysis of streaming social media data: Kinesis Data Streams -> Kinesis Data Analytics -> AWS Lambda -> DynamoDB
	- IoT analysis: Kinesis Data Streams -> AWS Lambda
	- Firehose is fully managed and only goes to S3, Redshift or ElasticSearch. Near real time
	- Streams is manually managed and can access other services. It can also store data (for up to 7 days). Real time
- *Many applications, especially in areas such as gaming, media, mobile apps, and financials, require very low latency for a great user experience. To improve the user experience, Global Accelerator directs user traffic to the application endpoint that is nearest to the client, which reduces internet latency and jitter*
- *AWS Global Accelerator and Amazon CloudFront are separate services that use the AWS global network and its edge locations around the world. CloudFront improves performance for both cacheable content (such as images and videos) and dynamic content (such as API acceleration and dynamic site delivery). Global Accelerator improves performance for a wide range of applications over TCP or UDP by proxying packets at the edge to applications running in one or more AWS Regions. Global Accelerator is a good fit for non-HTTP use cases, such as gaming (UDP), IoT (MQTT), or Voice over IP, as well as for HTTP use cases that specifically require static IP addresses or deterministic, fast regional failover. Both services integrate with AWS Shield for DDoS protection.*
- *ELB provides load balancing within one Region, AWS Global Accelerator provides traffic management across multiple Regions [...] AWS Global Accelerator complements ELB by extending these capabilities beyond a single AWS Region, allowing you to provision a global interface for your applications in any number of Regions. If you have workloads that cater to a global client base, we recommend that you use AWS Global Accelerator. If you have workloads hosted in a single AWS Region and used by clients in and around the same Region, you can use an Application Load Balancer or Network Load Balancer to manage your resources*



## Cost
- There are three ways to eliminate or at least reduce the extra traffic costs:
	- For S3 and DynamoDB, you can create a Gateway VPC Endpoint which is free and lets you communicate to S3 and DynamoDB from private subnets without natting (In private subnets, this was often done using NAT gateways which increase your traffic costs)
	- For some AWS services (e.g. Kinesis), you can create an Interface VPC Endpoint which is cheaper than a NAT gateway.
	- Run your workloads in public subnets and protect them with security groups.
- Usually non-critical workload -> use Spot instances


## Compute
- After creating an EC2 instance you cannot change its AZ (unless you create an AMI of that instance and launch a new instance from it)
- A placement group is a logical grouping of instances within a single Availability Zone
- Ways to trigger Lambda:
	- API Gateway event
	- DynamoDB event
	- S3 event
- AutoScaling termination policy: *Amazon EC2 Auto Scaling first identifies which of the two types (Spot or On-Demand) should be terminated. It then applies the termination policy in each Availability Zone individually, and identifies which instance (within the identified purchase option) in which Availability Zone to terminate that will result in the Availability Zones being most balanced. Then: Oldest launch configuration. Then: Instances closest to the next billing hour*


## Database
- It is best practice to enable automatic backups and set the backup window to occur during the daily low in write IOPS
- Indexed data - usually NoSQL
- *With IAM database authentication, you use an authentication token when you connect to your DB cluster. An authentication token is a string of characters that you use instead of a password. After you generate an authentication token, it's valid for 15 minutes before it expires. If you try to connect using an expired token, the connection request is denied.*
- To manage the keys used for encrypting and decrypting your Amazon Redshift resources, you use AWS Key Management Service (AWS KMS).
- *You use the reader endpoint for read-only connections for your Aurora cluster. This endpoint uses a load-balancing mechanism to help your cluster handle a query-intensive workload. The reader endpoint is the endpoint that you supply to applications that do reporting or other read-only operations on the cluster.*
- In Aurora you can read from the Multi-AZ **standby** instance (unlike RDS)
- Amazon RDS uses snapshots for backup. Snapshots are encrypted when created only if the database is encrypted and you can only select encryption for the database when you first create it. In this case the database, and hence the snapshots, ad unencrypted. However, you can create an encrypted copy of a snapshot. You can restore using that snapshot which creates a new DB instance that has encryption enabled. From that point on encryption will be enabled for all snapshots.
- Aurora Global Database replicates writes in the primary region with a typical latency of <1 second to secondary regions, for low latency global reads. In disaster recovery situations, you can promote a secondary region to take full read-write responsibilities in under a minute.
- Aurora Global Database introduces a higher level of failover capability than a default Aurora Cluster, Recovering Times: RPO 5 Seconds, RTO less than 1 minute
- DynamoDB - go for **auto scaling** for a predictable workload and **on-demand** for unpredictable workload
- Standby instance by definition means Multi-AZ (and not e.g. multi-region)


## Monitoring
- Use CloudWatch TunnelState Metric to monitor whether a managed VPN connection is up or down
- If you want to enable a view of all API events for all current and future regions - create a new CloudTrail trail and apply it to all regions. Once you enable Cloudtrail on a root account, it will log all API interactions on the account and it will also propagate automatically to any new region defined in the account
- CloudTrail is enabled by default
- AWS Key Management Service is integrated with AWS CloudTrail to provide you with logs of all key usage to help meet your regulatory and compliance needs
- CloudWatch - Since AWS does not have access to the instance at the OS level, only metrics that can be monitored through the Hypervisor layer (such as CPU and Network Utilization) are recorded. Use Agent + custom metric otherwise.
- There are 2 main types of monitoring you can do on AWS EC2 Instances:
	- Basic Monitoring for Amazon EC2 instances: Seven pre-selected metrics at five-minute frequency and three status check metrics at one-minute frequency, for no additional charge.
	- Detailed Monitoring for Amazon EC2 instances: All metrics available to Basic Monitoring at one-minute frequency, for an additional charge. Instances with Detailed Monitoring enabled allows data aggregation by Amazon EC2 AMI ID and instance type.
- *Elastic Load Balancing provides **access logs** that capture detailed information about requests sent to your load balancer. Each log contains information such as the time the request was received, the client's IP address, latencies, request paths, and server responses. You can use these access logs to analyze traffic patterns and troubleshoot issues.*




## Networking
- If you cannot RDP into the web server try the following:
	- Verify that you're using the correct public DNS hostname
	- Verify that your instance has a public IPv4 address (if not, add an Elastic IP Address)
	- To connect to your instance using an IPv6 address, check that your local computer has an IPv6 address and is configured to use IPv6
	- Verify that your security group has a rule that allows RDP access
	- Verify that the instance has passed status checks
	- Verify that Windows Firewall, or other firewall software, is not blocking RDP traffic to the instance
- Use VPC Endpoint to ensure EC2 <-> S3 traffic does not traverse internet
- NAT Gateway and NAT instance only support IPV4 traffic. For IPV6, you need to use egress only Internet Gateway
- AWS PrivateLink simplifies the security of data shared with cloud-based applications by eliminating the exposure of data to the public Internet. AWS PrivateLink provides private connectivity between VPCs, AWS services, and on-premises applications, securely on the Amazon network.
- *With SNI (Server Name Indication) support you can associate multiple certificates with a listener and each secure application behind a load balancer can use its own certificate. You can now host multiple secure (HTTPS) applications, each with its own SSL certificate, behind one load balancer.*
- *You can privately access AWS Systems Manager APIs from your VPC (created using Amazon Virtual Private Cloud) by creating VPC Endpoints. With VPC Endpoints, the routing between the VPC and AWS Systems Manager is handled by the AWS network without the need for an internet gateway, NAT gateway, or VPN connection. The latest generation of VPC Endpoints used by AWS Systems Manager are powered by AWS PrivateLink, a technology that enables private connectivity between AWS services using Elastic Network Interfaces (ENIs) with private IP addresses in your VPCs.*
- *Geolocation routing policy – Use when you want to route traffic based on the location of your users. Geoproximity routing policy – Use when you want to route traffic based on the location of your resources and, optionally, shift traffic from resources in one location to resources in another.*


## Scalability
- An Auto Scaling group cannot span multiple regions.


## Security
- In order to serve content that is stored in S3, but not publically accessible from S3 directly create a special CloudFront user called an origin access identity (OAI) and associate it with your distribution

- Use the root account to create your first IAM user and then lock away the root account. **Do not** use the root account only to create administrator accounts

- A presigned URL gives you access to the object identified in the URL, provided that the creator of the presigned URL has permissions to access that object. That is, if you receive a presigned URL to upload an object, you can upload the object only if the creator of the presigned URL has the necessary permissions to upload that object. When you create a presigned URL, you must provide your security credentials and then specify a bucket name, an object key, an HTTP method (PUT for uploading objects), and an expiration date and time

- Using IAM, you apply permissions to IAM users, groups, and roles by creating policies. You can create two types of IAM policies: Managed Policies and Inline Policies. Managed policies are standalone policies that you can attach to multiple users, groups, and roles in your AWS account. Managed policies apply only to identities (users, groups, and roles) - not resources. Inline policies are policies that you create and manage, and that are embedded directly into a single user, group, or role.

- If you want to allow the Lambda function to access the Amazon RDS database you need to create the Lambda function within the Amazon RDS VPC and change the ingress rules of RDS security group
- Use SSE-KMS for envelope encryption, key rotation and visibility into key usage
- Keys:
	- SSE-S3: AWS manages both data key and master key
	- SSE-KMS: AWS manages data key and you manage master key
	- SSE-C: You manage both data key and master key
- If Lambda function requires e.g. access to RDS DB you might configure it to use db passwords via AWS Systems Manager Parameter Store
- You can use WAF ONLY with:
	- CloudFront
	- ALB
	- API Gateway
- *A Lambda authorizer (formerly known as a custom authorizer) is an API Gateway feature that uses a Lambda function to control access to your API. A Lambda authorizer is useful if you want to implement a custom authorization scheme that uses a bearer token authentication strategy such as OAuth or SAML, or that uses request parameters to determine the caller's identity. When a client makes a request to one of your API's methods, API Gateway calls your Lambda authorizer, which takes the caller's identity as input and returns an IAM policy as output.*
- As an alternative to using IAM roles and policies or Lambda authorizers (formerly known as custom authorizers), you can use an Amazon Cognito user pool to control who can access your API in Amazon API Gateway.
- *KSM is shared hardware tenancy - you keys are in their own partition of an encryption module shared with other AWS customers, each with their own isolated partition. Cloud HSM gives you your own hardware module, so the most likely reason to choose Cloud HSM is if you had to ensure your keys were isolated on their own encryption module for compliance purposes. Also, AWS KSM only uses symmetric keys, while Cloud HSM allows symmetric and asymmetric keys.*



## Storage
- EBS snaphots are incremental. You only pay for storage (not the transfer)
- Use AWS Storage Gateway in stored mode if an application needs to interact with local storage
- If you want to copy backup EBS volumes to another region, create EBS snapshots and copy them to a desired region
- Sequential access -> choose HDD
- Use EBS if you're migrating a prioprietary file system
- EFS **is not** supported on Windows instances
- Efficient way to store images from a mobile app on S3 is to upload them to S3 using pre-signed URL
- Consider Raid 0 for increased performance
- Versioning is a pre-requisite for cross-region replication in S3
- Immutable data - use Glacier
- S3 with static website does not support HTTPS. It has to be enabled with cloudfront with S3 bucket as origin.
- Use AWS DataSync to migrate existing data to Amazon S3 vs Use File Gateway configuration of AWS Storage Gateway to retain access to the migrated data and for ONGOIN UPDATE from your on-premises file-based applications.


# Documentation links
[AutoScaling - Scaling Policy](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-as-policy.html)

[RAID Configuration on Linux](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/raid-config.html)

[Amazon Machine Images (AMI)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html)

[VPC endpoints](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-endpoints.html)

[Plan for Disaster Recovery (DR)](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/plan-for-disaster-recovery-dr.html)

[Security, Identity, and Compliance](https://docs.aws.amazon.com/whitepapers/latest/aws-overview/security-services.html)

[Implementing a disaster recovery strategy with Amazon RDS](https://aws.amazon.com/blogs/database/implementing-a-disaster-recovery-strategy-with-amazon-rds/)

[CloudTrail Concepts](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-concepts.html)

[RDS Multi-AZ](https://aws.amazon.com/rds/features/multi-az/)

[Amazon Kinesis](https://aws.amazon.com/kinesis/)
