export interface FAQ {
  id: number;
  question: string;
  answer: string;
  category: "billing" | "account" | "features" | "security" | "support";
  tags: string[];
}

export const FAQS: FAQ[] = [
  {
    id: 1,
    question: "How do I reset my password?",
    answer:
      "To reset your password, click 'Forgot Password' on the login page and enter your registered email address. You'll receive a reset link within 2 minutes. The link expires after 30 minutes. If you don't see the email, check your spam folder or contact support@nexaflow.io.",
    category: "account",
    tags: ["password", "reset", "login", "access"],
  },
  {
    id: 2,
    question: "What payment methods do you accept?",
    answer:
      "We accept all major credit and debit cards (Visa, Mastercard, Amex, Discover), PayPal, and bank transfers for annual enterprise plans. All payments are processed securely via Stripe. We do not store your card details on our servers.",
    category: "billing",
    tags: ["payment", "credit card", "billing", "stripe", "paypal"],
  },
  {
    id: 3,
    question: "Can I upgrade or downgrade my subscription plan?",
    answer:
      "Yes! You can change your plan at any time from Settings → Billing → Change Plan. Upgrades take effect immediately and are prorated. Downgrades apply at the start of your next billing cycle. No penalties or fees apply for plan changes.",
    category: "billing",
    tags: ["upgrade", "downgrade", "plan", "subscription", "change"],
  },
  {
    id: 4,
    question: "Is there a free trial available?",
    answer:
      "Absolutely. We offer a 14-day free trial on all Pro and Business plans — no credit card required. You get full access to all features. At the end of the trial you can choose a paid plan or automatically roll onto our Free tier (limited to 3 projects and 1 user).",
    category: "billing",
    tags: ["free", "trial", "demo", "test", "try"],
  },
  {
    id: 5,
    question: "How do I invite team members to my workspace?",
    answer:
      "Go to Settings → Team → Invite Members and enter the email addresses of your teammates. You can assign roles (Admin, Editor, Viewer) during the invite. Invitees will receive an email with a secure link. On the Free plan you can have 1 member; Pro supports up to 10, and Business is unlimited.",
    category: "features",
    tags: ["invite", "team", "members", "collaborate", "workspace", "users"],
  },
  {
    id: 6,
    question: "How do I cancel my subscription?",
    answer:
      "You can cancel anytime from Settings → Billing → Cancel Subscription. You'll retain access until the end of your current paid period. We don't offer partial-month refunds, but if you cancel within 7 days of a renewal charge, contact support for a full refund — no questions asked.",
    category: "billing",
    tags: ["cancel", "subscription", "refund", "stop", "end"],
  },
  {
    id: 7,
    question: "Is my data secure and encrypted?",
    answer:
      "Yes. All data is encrypted in transit using TLS 1.3 and at rest using AES-256. We are SOC 2 Type II certified, GDPR compliant, and undergo independent security audits quarterly. You can read our full Security Policy at nexaflow.io/security.",
    category: "security",
    tags: ["security", "encryption", "data", "safe", "gdpr", "privacy", "ssl"],
  },
  {
    id: 8,
    question: "Can I export my data?",
    answer:
      "Yes. You can export all your project data, reports, and account information at any time from Settings → Data → Export. We support CSV, JSON, and PDF formats. Enterprise customers also have access to our REST API for automated data exports.",
    category: "features",
    tags: ["export", "data", "download", "backup", "csv", "json"],
  },
  {
    id: 9,
    question: "What integrations do you support?",
    answer:
      "NexaFlow integrates natively with Slack, Google Workspace, Microsoft 365, Notion, Zapier, GitHub, Jira, HubSpot, and Salesforce. We also offer a REST API and webhooks so you can build custom integrations. See the full list at nexaflow.io/integrations.",
    category: "features",
    tags: ["integrations", "slack", "zapier", "api", "connect", "github", "jira"],
  },
  {
    id: 10,
    question: "How do I contact customer support?",
    answer:
      "You can reach us via live chat (bottom-right bubble, Mon–Fri 9am–6pm EST), email at support@nexaflow.io (response within 4 hours on business days), or through our Help Center at help.nexaflow.io. Enterprise customers have a dedicated Account Manager and 24/7 priority support.",
    category: "support",
    tags: ["support", "help", "contact", "chat", "email", "customer service"],
  },
  {
    id: 11,
    question: "What happens to my data if I cancel?",
    answer:
      "After cancellation, your data remains accessible in read-only mode for 30 days. After that, it is permanently deleted from our servers within 60 days in accordance with our data retention policy. We strongly recommend exporting your data before cancelling.",
    category: "account",
    tags: ["cancel", "data", "delete", "retention", "export", "account closure"],
  },
  {
    id: 12,
    question: "Do you offer discounts for nonprofits or students?",
    answer:
      "Yes! Verified nonprofit organizations receive a 50% discount on all plans. Students and educators with a valid .edu email get our Pro plan free for one year. Apply at nexaflow.io/discounts with supporting documentation.",
    category: "billing",
    tags: ["discount", "nonprofit", "student", "edu", "pricing", "deal"],
  },
  {
    id: 13,
    question: "Can I use NexaFlow on mobile devices?",
    answer:
      "Yes. NexaFlow has native iOS and Android apps available on the App Store and Google Play. The mobile apps support full project management, real-time collaboration, notifications, and offline mode. Your data syncs instantly across all devices.",
    category: "features",
    tags: ["mobile", "ios", "android", "app", "phone", "tablet", "offline"],
  },
  {
    id: 14,
    question: "How many projects can I create?",
    answer:
      "Free plan: up to 3 active projects. Pro plan: up to 50 projects. Business and Enterprise plans: unlimited projects. Archived projects don't count toward your limit. You can archive projects from the project settings menu.",
    category: "features",
    tags: ["projects", "limit", "how many", "create", "quota"],
  },
  {
    id: 15,
    question: "Is there an API I can use to build on top of NexaFlow?",
    answer:
      "Yes. Our REST API and GraphQL API are available on Pro and higher plans. Documentation is at api.nexaflow.io. You'll find endpoints for projects, tasks, users, reports, and webhooks. API keys are generated from Settings → Developer → API Keys. Rate limits: 1,000 req/min on Pro, 10,000 on Business, unlimited on Enterprise.",
    category: "features",
    tags: ["api", "developer", "rest", "graphql", "integrate", "build", "webhook"],
  },
];

export const SUGGESTED_QUESTIONS = [
  "How do I reset my password?",
  "Is there a free trial?",
  "What integrations do you support?",
  "How do I invite team members?",
  "Is my data secure?",
];

export const CATEGORY_COLORS: Record<FAQ["category"], string> = {
  billing: "#F59E0B",
  account: "#818CF8",
  features: "#34D399",
  security: "#F87171",
  support: "#60A5FA",
};
